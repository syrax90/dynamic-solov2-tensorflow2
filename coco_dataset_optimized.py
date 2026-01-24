"""
Author: Pavel Timonin
Created: 2025-10-17
Description: This script contains classes and functions of COCO dataset optimized for tf.Dataset.
"""


import os
from typing import Optional, Tuple
from coco_dataset import compute_scale_ranges

import tensorflow as tf

# Feature spec that matches the COCO TFRecord format
_FEATURES = {
    # image-level fields
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/filename": tf.io.FixedLenFeature([], tf.string),
    "image/id": tf.io.FixedLenFeature([], tf.int64),
    "image/format": tf.io.FixedLenFeature([], tf.string),

    # per-object fields
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/area": tf.io.VarLenFeature(tf.float32),
    "image/object/category_id": tf.io.VarLenFeature(tf.int64),
    "image/object/iscrowd": tf.io.VarLenFeature(tf.int64),
    "image/object/mask_png": tf.io.VarLenFeature(tf.string),
}

@tf.function
def process_one(
        cate_grid, mask_tensor, seg_mask_orig, bbox, cat_id,
        height_scale, width_scale, cell_h, cell_w,
        grid_size_t, feat_h_t, feat_w_t, center_radius_t, scale_min_t, scale_max_t
):
    """
    Assigns a category to a neighborhood of SOLO grid cells around an object's center
    and writes the object's binary mask into the corresponding mask channels.

    This op:
      1) Filters the object by its scale using sqrt(area) ∈ [scale_min_t, scale_max_t).
      2) Computes the center of mass (COM) of `seg_mask_orig` in the original image space.
      3) Maps the COM to the feature-map (FPN) grid and selects a (2r+1)×(2r+1) neighborhood
         of grid cells around it (where r = `center_radius_t`).
      4) For cells that are currently unassigned (value == -1), sets the category to `cat_id`
         in `cate_grid` and writes the resized binary mask into the corresponding mask
         channels of `mask_tensor`.

    Args:
        cate_grid (tf.Tensor): Shape `[S, S]`, dtype `tf.int32`. Category target grid for a
            single scale. Unassigned cells must contain `-1`.
        mask_tensor (tf.Tensor): Shape `[H, W, S*S]`, dtype `tf.uint8`. Per-cell binary masks
            for the same SOLO grid. Channel `c = gy * S + gx` corresponds to grid cell `(gy, gx)`.
        seg_mask_orig (tf.Tensor): Shape `[img_h, img_w]` (or `[img_h, img_w, 1]`), integer/bool.
            Foreground is values `> 0`. Used to compute the COM and as the source for the mask.
        bbox (tf.Tensor): Shape `[4]`, dtype `tf.float32`. Bounding box `[x, y, w, h]` in the
            original image coordinate space; used only to compute `area = w*h` for scale filtering.
        cat_id (tf.Tensor or int): Scalar `int32`. The category ID to assign to eligible cells.
        height_scale (tf.Tensor or float): Scalar `float32`. Factor to map original `y` to FPN `y`
            (typically `feat_h / img_h`).
        width_scale (tf.Tensor or float): Scalar `float32`. Factor to map original `x` to FPN `x`
            (typically `feat_w / img_w`).
        cell_h (tf.Tensor or float): Scalar `float32`. Height of one SOLO cell in FPN pixels.
        cell_w (tf.Tensor or float): Scalar `float32`. Width of one SOLO cell in FPN pixels.
        grid_size_t (tf.Tensor or int): Scalar `int32`. SOLO grid size `S`.
        feat_h_t (tf.Tensor or int): Scalar `int32`. Feature-map height `H` for mask resizing.
        feat_w_t (tf.Tensor or int): Scalar `int32`. Feature-map width `W` for mask resizing.
        center_radius_t (tf.Tensor or int): Scalar `int32`. Radius `r` of the neighborhood (in cells)
            around the COM to consider: offsets in `[-r, r]` for both x and y.
        scale_min_t (tf.Tensor or float): Scalar `float32`. Minimum allowed `sqrt(area)` (inclusive).
        scale_max_t (tf.Tensor or float): Scalar `float32`. Maximum allowed `sqrt(area)` (exclusive).

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - cate_grid_out (tf.Tensor): Shape `[S, S]`, dtype `tf.int32`. Updated category grid.
            - mask_tensor_out (tf.Tensor): Shape `[H, W, S*S]`, dtype `tf.uint8`. Updated mask tensor
              with the resized `seg_mask_orig` written into channels for newly assigned cells.

    Behavior:
        - If `sqrt(area)` is outside `[scale_min_t, scale_max_t)`, inputs are returned unchanged.
        - If `seg_mask_orig` has no foreground pixels, inputs are returned unchanged.
        - Only cells whose current value is `-1` are updated; pre-assigned cells are left intact.
        - Masks are resized to `[feat_h_t, feat_w_t]` using nearest-neighbor interpolation.
        - For each newly assigned cell `(gy, gx)`, the mask is written into channel
          `c = gy * S + gx`.

    Notes:
        - `mask_tensor` must be `tf.uint8` to match the resized binary mask produced here.
        - Coordinate mapping:
            COM (orig) → FPN: `x_fpn = x_mean * width_scale`, `y_fpn = y_mean * height_scale`
            Center cell indices: `gx = floor(x_fpn / cell_w)`, `gy = floor(y_fpn / cell_h)`,
            each clipped to `[0, S-1]`.
    """
    # Unpack bbox
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    # Filter by area/scale range
    area = w * h
    def _skip():
        return cate_grid, mask_tensor

    def _go():
        # Center of mass (original space) from mask > 0
        coords = tf.where(seg_mask_orig > 0)
        n_pts = tf.shape(coords)[0]
        def _empty():
            return cate_grid, mask_tensor
        def _non_empty():
            # y,x means row,col
            y_mean = tf.reduce_mean(tf.cast(coords[:, 0], tf.float32))
            x_mean = tf.reduce_mean(tf.cast(coords[:, 1], tf.float32))

            # Map COM to FPN space
            x_fpn = x_mean * width_scale
            y_fpn = y_mean * height_scale

            # Center cell indices
            gx_center = tf.cast(tf.math.floor(x_fpn / cell_w), tf.int32)
            gy_center = tf.cast(tf.math.floor(y_fpn / cell_h), tf.int32)
            gx_center = tf.clip_by_value(gx_center, 0, grid_size_t - 1)
            gy_center = tf.clip_by_value(gy_center, 0, grid_size_t - 1)

            # Resize mask to feature map (nearest)
            # Ensure shape [H,W]
            seg = tf.cast(seg_mask_orig > 0, tf.uint8)
            seg = tf.image.resize(
                tf.expand_dims(seg, axis=-1),  # [H0,W0,1]
                size=(feat_h_t, feat_w_t),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                preserve_aspect_ratio=False,
                antialias=False
            )
            seg = tf.squeeze(tf.cast(seg, tf.uint8), axis=-1)  # [H,W]

            # Neighborhood offsets in a vectorized way
            r = center_radius_t
            rng = tf.range(-r, r + 1, dtype=tf.int32)
            dx, dy = tf.meshgrid(rng, rng, indexing='xy')
            dx = tf.reshape(dx, [-1])  # [K]
            dy = tf.reshape(dy, [-1])  # [K]
            gx = tf.clip_by_value(gx_center + dx, 0, grid_size_t - 1)
            gy = tf.clip_by_value(gy_center + dy, 0, grid_size_t - 1)
            neigh_indices = tf.stack([gy, gx], axis=-1)  # [K,2]

            # Current values at neighbor cells
            current_vals = tf.gather_nd(cate_grid, neigh_indices)  # [K]
            is_free = tf.equal(current_vals, -1)

            # Cells we will actually assign (only the free ones)
            # Build updates for category grid
            k = tf.shape(neigh_indices)[0]
            cat_updates = tf.fill([k], cat_id)
            # But only for free cells; otherwise keep current
            final_updates = tf.where(is_free, cat_updates, current_vals)

            # Update cate_grid via scatter
            new_cate_grid = tf.tensor_scatter_nd_update(
                cate_grid, neigh_indices, final_updates
            )

            # Mask updates only for newly assigned channels
            new_cells_indices = tf.boolean_mask(neigh_indices, is_free)  # [K_free,2]
            new_channels = new_cells_indices[:, 0] * grid_size_t + new_cells_indices[:, 1]  # [K_free]
            k_free = tf.shape(new_channels)[0]

            def _update_masks():
                # Make a channel boolean mask [C] with True for channels to update
                onehot = tf.one_hot(new_channels, depth=grid_size_t * grid_size_t, dtype=tf.int32)
                onehot = tf.cast(onehot, tf.bool)
                ch_mask = tf.reduce_any(onehot, axis=0)  # [C] bool

                # Broadcast channel mask to [H,W,C]
                ch_mask_b = tf.reshape(ch_mask, [1, 1, -1])
                ch_mask_b = tf.broadcast_to(ch_mask_b, tf.shape(mask_tensor))

                # Broadcast seg to [H,W,1]
                seg_b = tf.expand_dims(seg, axis=-1)
                seg_b = tf.broadcast_to(seg_b, tf.shape(mask_tensor))

                # Replace only at target channels
                updated_mask = tf.where(ch_mask_b, seg_b, mask_tensor)
                return updated_mask

            new_mask_tensor = tf.cond(
                tf.greater(k_free, 0),
                true_fn=_update_masks,
                false_fn=lambda: mask_tensor
            )

            return new_cate_grid, new_mask_tensor

        return tf.cond(tf.equal(n_pts, 0), _empty, _non_empty)

    in_scale = tf.logical_and(
        tf.greater(area, 0.0),
        tf.logical_and(
            tf.greater_equal(tf.sqrt(area), scale_min_t),
            tf.less(tf.sqrt(area), scale_max_t)
        )
    )
    return tf.cond(in_scale, _go, _skip)

@tf.function
def generate_solo_targets_single_scale(
    categories,
    resized_masks,
    resized_bboxes,
    target_height,
    target_width,
    grid_size,
    scale_range,     # (scale_min, scale_max) for sqrt(area), e.g. (0, 64)
    scale=0.25,      # scale factor for feature map size (default 1/4)
    center_radius=0  # radius (in grid cells) around center to assign
):
    """
    Builds SOLO targets (category grid and per-cell masks) for one grid scale.

    Given per-instance categories, masks, and boxes already resized to the target
    training dimensions, this function:
      * Computes the feature-map size from `scale`.
      * Initializes empty targets (category grid filled with -1, mask channels 0).
      * Iterates over instances and calls `process_one` to assign categories
        and write mask channels within a neighborhood of the instance center.
      * Returns per-scale targets.

    Args:
      categories: Tensor [N] (int32). Category id per instance.
      resized_masks: Tensor [N, Ht, Wt] (uint8). Resized binary masks aligned to
        `target_height` x `target_width`. Values are 0/1 (or 0/255 then cast).
      resized_bboxes: Tensor [N, 4] (float32). Boxes in (x, y, w, h) format
        aligned to the same target space as `resized_masks`.
      target_height: Scalar int32. Target image height (Ht).
      target_width: Scalar int32. Target image width (Wt).
      grid_size: Scalar int32. SOLO grid size S for this scale.
      scale_range: Tensor/tuple [2] (float32). (min_sqrt_area, max_sqrt_area)
        of instances to include at this scale.
      scale: Scalar float (default 0.25). Feature-map scale relative to target
        image size, used to compute feature resolution (Hf, Wf).
      center_radius: Scalar int32 (default 0). Neighborhood radius in grid cells.

    Returns:
      Tuple[Tensor, Tensor]:
        - cate_target: Tensor [S, S] (int32). Category grid with -1 for empty cells.
        - mask_target: Tensor [Hf, Wf, S*S] (uint8). Concatenated per-cell masks.
    """
    scale_min = scale_range[0]
    scale_max = scale_range[1]

    # Compute feature map shape from scale
    feat_h = tf.cast(tf.round(tf.cast(target_height, tf.float32) * scale), tf.int32)
    feat_w = tf.cast(tf.round(tf.cast(target_width, tf.float32) * scale), tf.int32)

    # Prepare empty targets
    cate_target = tf.cast(tf.fill([grid_size, grid_size], -1), tf.int32)
    mask_channels = grid_size * grid_size
    mask_target = tf.zeros([feat_h, feat_w, mask_channels], dtype=tf.uint8)

    # Precompute scales and cell sizes (as TF scalars)
    cell_h = tf.cast(feat_h, tf.float32) / tf.cast(grid_size, tf.float32)
    cell_w = tf.cast(feat_w, tf.float32) / tf.cast(grid_size, tf.float32)

    def body(i, cate_target_changed, mask_target_changed):
        cate_id = categories[i]
        mask = resized_masks[i]
        bbox = resized_bboxes[i]

        cate_target_result, mask_target_result = process_one(cate_target_changed, mask_target_changed, mask, bbox,
                                                             cate_id, scale, scale, cell_h, cell_w, grid_size, feat_h,
                                                             feat_w, center_radius, scale_min, scale_max)

        return i + 1, cate_target_result, mask_target_result

    i0 = tf.constant(0)
    i_f, cate_target, mask_target = tf.while_loop(
        lambda i, c, m : i < tf.shape(categories)[0],
        body,
        loop_vars=[i0, cate_target, mask_target],
    )

    return cate_target, mask_target

def sparse_to_dense_1d(v, dtype):
    """
    Convert a VarLen sparse tensor to a 1D dense tensor (length N).

    Args:
      v: `tf.SparseTensor`. A rank-1 sparse tensor.
      dtype: `tf.DType`. The desired dtype of the output.

    Returns:
      Tensor: Dense 1D tensor with shape [N] and dtype `dtype`.
    """
    return tf.cast(tf.sparse.to_dense(v), dtype)

# -----------------------------
# Data augmentations (graph mode, TF-only)
# -----------------------------
def maybe_hflip(img, masks, bboxes):
    """
    Randomly applies a horizontal flip to image, masks, and boxes (p=0.5).

    Args:
      img: Tensor [H, W, C] (uint8). Image.
      masks: Tensor [N, H, W] (uint8). Per-instance binary masks aligned to `img`.
      bboxes: Tensor [N, 4] (float32). Boxes in (x, y, w, h) format in the same
        coordinate space as `img`.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - img_f: Tensor [H, W, C] (uint8). Possibly flipped image.
        - masks_f: Tensor [N, H, W] (uint8). Possibly flipped masks.
        - b_new: Tensor [N, 4] (float32). Updated boxes after flip.

    Notes:
      * Applies with probability 0.5.
      * Boxes are mirrored around the image center by updating x: `x' = W - x - w`.
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        # Flip image and masks
        img_f = tf.image.flip_left_right(img)
        masks_f = tf.reverse(masks, axis=[2])  # [N,H,W], flip width

        # Adjust boxes
        W = tf.cast(tf.shape(img)[1], tf.float32)
        x, y, bw, bh = tf.unstack(bboxes, axis=1)
        x_new = W - x - bw
        b_new = tf.stack([x_new, y, bw, bh], axis=1)
        return img_f, masks_f, b_new
    def no():
        return img, masks, bboxes
    return tf.cond(do, yes, no)

def maybe_brightness(img):
    """
    Randomly jitters brightness by a multiplicative factor in [-20%, +20%] (p=0.5).

    Args:
      img: Tensor [H, W, C] (uint8). Image in range [0, 255].

    Returns:
      Tensor: Image of shape [H, W, C] (uint8) with brightness possibly adjusted.

    Notes:
      * Applies with probability 0.5.
      * The factor is sampled uniformly from [0.8, 1.2] and values are clipped to [0, 255].
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        factor = 1.0 + (tf.random.uniform([], -0.2, 0.2))
        img_f32 = tf.cast(img, tf.float32) * factor
        img_f32 = tf.clip_by_value(img_f32, 0.0, 255.0)
        return tf.cast(img_f32, tf.uint8)
    def no():
        return img
    return tf.cond(do, yes, no)

def maybe_scale(img, masks, bboxes):
    """
    Randomly scales image, masks, and boxes uniformly (p=0.5).

    The scale factor `s` is sampled from [0.8, 1.2]. Images are resized with bilinear
    interpolation; masks use nearest neighbor. Boxes are scaled by `s`.

    Args:
      img: Tensor [H, W, C] (uint8). Input image.
      masks: Tensor [N, H, W] (uint8). Per-instance masks aligned with `img`.
      bboxes: Tensor [N, 4] (float32). Boxes (x, y, w, h) in `img` coordinates.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - img_rs: Tensor [⌊H*s⌉, ⌊W*s⌉, C] (uint8). Resized image.
        - masks_rs: Tensor [N, ⌊H*s⌉, ⌊W*s⌉] (uint8). Resized masks.
        - b_new: Tensor [N, 4] (float32). Scaled boxes.

    Notes:
      * Applies with probability 0.5.
      * Image values are clipped to [0, 255] after bilinear resize and rounding.
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        s = tf.random.uniform([], 0.8, 1.2)

        # New size
        orig_hw = tf.cast(tf.shape(img)[:2], tf.float32)  # [H, W]
        new_hw = tf.cast(tf.round(orig_hw * s), tf.int32)
        new_h = new_hw[0]
        new_w = new_hw[1]

        # Resize image (bilinear) and masks (nearest)
        img_f32 = tf.cast(img, tf.float32)
        img_rs  = tf.image.resize(img_f32, size=[new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
        img_rs  = tf.clip_by_value(img_rs, 0.0, 255.0)
        img_rs  = tf.cast(tf.round(img_rs), tf.uint8)

        # Resize all masks at once
        masks_ch = tf.expand_dims(masks, axis=-1)                         # [N,H,W,1]
        masks_rs = tf.image.resize(masks_ch, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        masks_rs = tf.squeeze(masks_rs, axis=-1)                             # [N,H,W]
        masks_rs = tf.cast(masks_rs, tf.uint8)

        # Scale boxes and areas
        b_new = bboxes * tf.stack([s, s, s, s])  # (x,y,w,h) * s

        return img_rs, masks_rs, b_new
    def no():
        return img, masks, bboxes
    return tf.cond(do, yes, no)

def maybe_random_crop(img, masks, bboxes, cat_ids):
    """
    Applies a random crop up to 20% per side, updating masks/boxes/categories (p=0.5).

    A rectangular crop is sampled with left/top margins in [0, 0.2*W/H] and
    right/bottom margins in [0, 0.2*W/H]. Boxes are translated to the crop
    coordinate frame, clipped, and instances with very small resulting boxes
    (<=1 px width or height) are removed. The function keeps `cat_ids` and `masks`
    aligned with the filtered boxes.

    Args:
      img: Tensor [H, W, C] (uint8). Input image.
      masks: Tensor [N, H, W] (uint8). Per-instance masks.
      bboxes: Tensor [N, 4] (float32). Boxes (x, y, w, h) in image coords.
      cat_ids: Tensor [N] (int32). Category id per instance.

    Returns:
      Tuple[Tensor, Tensor, Tensor, Tensor]:
        - img_cr: Tensor [Hc, Wc, C] (uint8). Cropped image.
        - m_new: Tensor [N', Hc, Wc] (uint8). Cropped masks for kept instances.
        - b_new: Tensor [N', 4] (float32). Boxes translated/clipped to the crop.
        - c_new: Tensor [N'] (int32). Category ids for kept instances.

    Notes:
      * Applies with probability 0.5.
      * Keeps only boxes with width > 1 and height > 1 in pixels after cropping.
    """
    do = tf.less(tf.random.uniform([], 0, 1.0), 0.5)
    def yes():
        H = tf.shape(img)[0]
        W = tf.shape(img)[1]
        Hf = tf.cast(H, tf.float32)
        Wf = tf.cast(W, tf.float32)

        max_crop_x = tf.cast(tf.floor(Wf * 0.2), tf.int32)
        max_crop_y = tf.cast(tf.floor(Hf * 0.2), tf.int32)

        # Sample crop bounds: ensure left <= right and top <= bottom
        x1 = tf.random.uniform([], minval=0,              maxval=max_crop_x + 1, dtype=tf.int32)
        y1 = tf.random.uniform([], minval=0,              maxval=max_crop_y + 1, dtype=tf.int32)
        x2 = tf.random.uniform([], minval=W - max_crop_x, maxval=W + 1,         dtype=tf.int32)
        y2 = tf.random.uniform([], minval=H - max_crop_y, maxval=H + 1,         dtype=tf.int32)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Crop image and masks
        img_cr = tf.slice(img,   [y1, x1, 0], [crop_h, crop_w, -1])
        masks_cr = tf.slice(masks, [0, y1, x1], [-1, crop_h, crop_w])  # [N, crop_h, crop_w]

        # Adjust boxes to crop region, clip, and filter
        x, y, bw, bh = tf.unstack(bboxes, axis=1)
        x1f = tf.cast(x1, tf.float32)
        y1f = tf.cast(y1, tf.float32)
        cwf = tf.cast(crop_w, tf.float32)
        chf = tf.cast(crop_h, tf.float32)

        nx = tf.maximum(0.0, x - x1f)
        ny = tf.maximum(0.0, y - y1f)
        nw = tf.maximum(0.0, tf.minimum(bw, cwf - nx))
        nh = tf.maximum(0.0, tf.minimum(bh, chf - ny))

        keep = tf.logical_and(nw > 1.0, nh > 1.0)  # discard tiny/invalid boxes

        # Apply mask to per-instance tensors
        b_new = tf.boolean_mask(tf.stack([nx, ny, nw, nh], axis=1), keep)
        c_new = tf.boolean_mask(cat_ids,   keep)
        m_new = tf.boolean_mask(masks_cr,  keep, axis=0)


        return img_cr, m_new, b_new, c_new
    def no():
        return img, masks, bboxes, cat_ids
    return tf.cond(do, yes, no)


def parse_example(
        serialized,
        target_height,
        target_width,
        grid_sizes,
        scale,
        scale_ranges,
        augment):
    """
    Parses one TFRecord example and builds multi-scale SOLO training targets.

    This function:
      * Parses a single serialized example using `_FEATURES`.
      * Decodes the image (to RGB if needed) and per-instance masks (PNG).
      * Optionally applies augmentations (flip, brightness, random crop).
      * Resizes image to `(target_height, target_width)` and masks via nearest.
      * Scales boxes to the resized image coordinate frame.
      * For each SOLO grid size and its corresponding `scale_range`, generates
        per-scale targets via `generate_solo_targets_single_scale`, then
        concatenates category targets (flattened per scale) and mask targets
        (concatenated along channel axis).

    Args:
      serialized: Scalar string Tensor. A single serialized `tf.train.Example`.
      target_height: Scalar int. Output image height.
      target_width: Scalar int. Output image width.
      grid_sizes: Tensor/array-like [num_scales] (int32). SOLO grid sizes per scale.
      scale: Scalar float. Feature-map downscale (e.g., 0.25 -> 1/4).
      scale_ranges: Tensor [num_scales, 2] (float32). Per-scale (min, max) for
        sqrt(area) gating.
      augment: Scalar bool. If True, apply data augmentations.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - image_resized: Tensor [target_height, target_width, 3] (float32) in [0, 1].
        - cate_targets: Tensor [sum(S_i^2)] (int32). Concatenated category
          targets from all scales, flattened per scale then concatenated.
        - mask_targets: Tensor [Hf, Wf, sum(S_i^2)] (uint8). Concatenated
          per-cell masks across all scales. Hf/Wf match the feature size for the
          provided `scale`.

    Notes:
      * If there are zero instances, shapes are still well-defined: category
        targets will be a concatenation of -1-filled grids; mask targets will be
        all zeros.
    """

    ex = tf.io.parse_single_example(serialized, _FEATURES)

    # Scalars
    height  = tf.cast(ex["image/height"],  tf.int32)
    width   = tf.cast(ex["image/width"],   tf.int32)
    img_enc = ex["image/encoded"]                  # bytes

    scale_y = tf.cast(target_height, tf.float32) / tf.cast(height, tf.float32)
    scale_x = tf.cast(target_width, tf.float32) / tf.cast(width, tf.float32)

    # Variable-length (per-object) -> dense 1D tensors
    xmin = sparse_to_dense_1d(ex["image/object/bbox/xmin"], tf.float32)
    ymin = sparse_to_dense_1d(ex["image/object/bbox/ymin"], tf.float32)
    xmax = sparse_to_dense_1d(ex["image/object/bbox/xmax"], tf.float32)
    ymax = sparse_to_dense_1d(ex["image/object/bbox/ymax"], tf.float32)

    x = xmin
    y = ymin
    w = (xmax - xmin)
    h = (ymax - ymin)

    cat_ids   = sparse_to_dense_1d(ex["image/object/category_id"], tf.int32)
    mask_pngs = sparse_to_dense_1d(ex["image/object/mask_png"], tf.string)

    # Stack boxes as [N, 4] in (x, y, w, h) format
    bboxes = tf.stack([x, y, w, h], axis=1) if tf.size(xmin) > 0 else tf.zeros([0, 4], tf.float32)

    # Decode image to ensure 3 channels
    img = tf.io.decode_image(img_enc, expand_animations=False)  # uint8, shape [H,W,C]
    # Ensure we have 3 channels (COCO images should be RGB)
    img = tf.cond(tf.shape(img)[-1] == 3,
                  lambda: img,
                  lambda: tf.image.grayscale_to_rgb(img))

    # Decode each per-object PNG into [H, W] uint8;
    def _decode_one(png_bytes):
        m = tf.io.decode_png(png_bytes, channels=1)  # [H,W,1]
        return tf.squeeze(m, axis=-1)  # [H,W]

    masks = tf.map_fn(_decode_one, mask_pngs, fn_output_signature=tf.uint8) # shape [N, H, W]

    def _apply_augmentation():
        # Horizontal flip
        img_aug, masks_aug, bboxes_aug = maybe_hflip(img, masks, bboxes)

        # Brightness jitter (+/-20%)
        img_aug = maybe_brightness(img_aug)

        # Random scaling (0.8x–1.2x)
        #img_aug, masks_aug, bboxes_aug = maybe_scale(img_aug, masks_aug, bboxes_aug)

        # Random crop (≤20% each side); updates and filters instance-aligned tensors
        img_aug, masks_aug, bboxes_aug, cat_ids_aug = maybe_random_crop(
            img_aug, masks_aug, bboxes_aug, cat_ids
        )

        return img_aug, masks_aug, bboxes_aug, cat_ids_aug

    img, masks, bboxes, cat_ids = tf.cond(augment, _apply_augmentation, lambda: (img, masks, bboxes, cat_ids))

    # Resize (bilinear by default; set method if you need a match)
    image_resized = tf.image.resize(img, size=(target_height, target_width), method="bilinear", antialias=True)

    # Convert to float32 in [0, 1]
    image_resized = tf.cast(image_resized, tf.float32) / 255.0

    # resize masks to (target_height, target_width) using nearest neighbor ===
    # Vectorized resize over batch dimension; works even if N == 0.
    masks_resized = tf.image.resize(
        tf.expand_dims(tf.cast(masks, tf.float32), axis=-1),  # [N,H,W,1]
        size=(target_height, target_width),
        method="nearest"
    )
    masks_resized = tf.cast(tf.round(tf.squeeze(masks_resized, axis=-1)), tf.uint8)  # [N,new_h,new_w]

    scales = tf.cast([scale_x, scale_y, scale_x, scale_y], dtype=tf.float32)
    bboxes = bboxes * scales  # element-wise broadcast

    cat_ids = cat_ids - 1  # Convert to 0-based category ids

    def _generate_solo_targets_multi_scale(i, cate_acc, mask_acc):
        # enumerate i-th grid/scale_range
        grid_size_i = grid_sizes[i]
        scale_range_i = scale_ranges[i]  # shape [2], e.g. (scale_min, scale_max)

        # Compute single-scale targets
        cate_target_i, mask_target_i = generate_solo_targets_single_scale(
            cat_ids,
            masks_resized,
            bboxes,
            target_height,
            target_width,
            grid_size_i,
            scale_range_i,  # (scale_min, scale_max) for sqrt(area)
            scale,  # feature-map scale factor (e.g., 1/4)
            center_radius=0  # radius (grid cells) around center to assign
        )

        # Flatten category target from [grid_size, grid_size] -> [grid_size*grid_size]
        cate_target_i = tf.reshape(cate_target_i, [-1])

        new_cate_acc = tf.concat([cate_acc, cate_target_i], axis=0)
        new_mask_acc = tf.concat([mask_acc, mask_target_i], axis=2)

        return i + 1, new_cate_acc, new_mask_acc

    # Number of scales
    num_scales = tf.shape(grid_sizes)[0]

    feat_h = tf.cast(tf.round(tf.cast(target_height, tf.float32) * scale), tf.int32)
    feat_w = tf.cast(tf.round(tf.cast(target_width, tf.float32) * scale), tf.int32)

    # Initialize accumulators
    cate_init = tf.zeros([0], dtype=tf.int32)
    mask_init = tf.zeros(
        [tf.cast(feat_h, tf.int32),
         tf.cast(feat_w, tf.int32),
         0],
        dtype=tf.uint8
    )

    # Loop
    _, cate_targets, mask_targets = tf.while_loop(
        cond=lambda i, *_: i < num_scales,
        body=_generate_solo_targets_multi_scale,
        loop_vars=[tf.constant(0), cate_init, mask_init],
        shape_invariants=[
            tf.TensorShape([]),  # i
            tf.TensorShape([None]),  # cate_acc grows along axis 0
            tf.TensorShape([None, None, None])  # mask_acc grows along axis 2
        ]
    )

    return image_resized, cate_targets, mask_targets

def create_coco_tfrecord_dataset(
    train_tfrecord_directory: str,
    target_size: Tuple[int, int],
    batch_size: int,
    grid_sizes,
    scale: float = 2.5,
    deterministic: bool = False,
    augment: bool = True,
    shuffle_buffer_size: Optional[int] = None,
    number_images: Optional[int] = None
) -> tf.data.Dataset:
    """Creates a `tf.data.Dataset` from COCO TFRecord shards and emits SOLO targets.

    This utility:
      * Scans a directory for `*.tfrecord` shards.
      * Builds a streaming `TFRecordDataset`.
      * Optionally shuffles and/or limits the number of examples.
      * Parses each example and constructs multi-scale SOLO targets via `parse_example`.
      * Batches and prefetches the dataset.

    Args:
      train_tfrecord_directory: Path to directory containing TFRecord shards.
      target_size: Tuple[int, int]. Target (height, width) for image & mask resizing.
      batch_size: Batch size for the resulting dataset.
      grid_sizes: Sequence of ints. SOLO grid sizes per scale (e.g., [40, 36, ...]).
      scale: Float. Feature-map scale factor used in target generation (e.g., 2.5).
        Note: This is later passed to `parse_example` which expects a downscale
        factor (e.g., 0.25); ensure consistency with your pipeline.
      deterministic: If False (default), allow non-deterministic map parallelism.
      augment: If True (default), apply data augmentations in `parse_example`.
      shuffle_buffer_size: Optional shuffle buffer size. If provided, shuffling is enabled.
      number_images: Optional cap on the number of images to take from the stream.

    Returns:
      tf.data.Dataset: A dataset of batched tuples:
        - image_resized: [B, Ht, Wt, 3] (float32) in [0, 1]
        - cate_targets: [B, sum(S_i^2)] (int32)
        - mask_targets: [B, Hf, Wf, sum(S_i^2)] (uint8)
    """
    scale_ranges = compute_scale_ranges(target_size[0], target_size[1], num_levels=len(grid_sizes))
    scale_ranges = tf.stack(scale_ranges, axis=0)
    scale_ranges = tf.cast(scale_ranges, tf.float32)

    grid_sizes_tf = tf.cast(grid_sizes, tf.int32)

    target_height, target_width = target_size
    augment_tf = tf.constant(augment)

    # Gather all shard paths (common suffixes)
    pattern = "*.tfrecord"
    files = tf.io.gfile.glob(os.path.join(train_tfrecord_directory, pattern))

    if not files:
        raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecord_directory}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))

    # Shuffle
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    if number_images is not None:
        ds = ds.take(number_images)

    # Parse
    ds = ds.map(lambda x: parse_example(x, target_height=target_height, target_width=target_width,
                                        grid_sizes=grid_sizes_tf, scale=scale, scale_ranges=scale_ranges, augment=augment_tf),
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
