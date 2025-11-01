"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script performs the model testing process.
"""

import cv2
import os
import random

import keras.models
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import DynamicSOLOConfig
from coco_dataset import get_classes
from model_functions import SOLOModel
import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def preprocess_image(image_path, input_size=(320, 320)):
    """
    Load an image, resize to a fixed size, and normalize pixel values to [0, 1].

    Also returns the original image shape and the unnormalized RGB image.

    Args:
      image_path (str): Path to the input image file.
      input_size (Tuple[int, int], optional): Target size `(width, height)` to
        resize the image to. Defaults to `(320, 320)`.

    Returns:
      Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
        A 3-tuple `(img_resized, original_shape, img)` where:
          * `img_resized` (np.ndarray): Resized and normalized RGB image of
            shape `(H, W, 3)`, dtype `float32`, values in `[0, 1]`.
          * `original_shape` (Tuple[int, int]): Original image height and width
            as `(H_orig, W_orig)`.
          * `img` (np.ndarray): Original RGB image **before** resizing and
            normalization, dtype `uint8`, values in `[0, 255]`.

    Raises:
      ValueError: If the image cannot be read from `image_path`.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]  # (H, W)

    # Resize
    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32) / 255.0

    return img_resized, original_shape, img

@tf.function
def matrix_nms(masks, scores, labels, pre_nms_k=500, post_nms_k=100, score_threshold=0.5, sigma=0.5):
    """
    Perform Matrix NMS for class-aware suppression of instance masks.

    This implementation follows the Matrix NMS formulation where each candidate’s
    score is decayed by the maximum IoU it has with higher-scoring, same-class
    candidates. Masks are first binarized at 0.5.

    Args:
      masks (tf.Tensor): Predicted masks of shape `(N, H, W)`, values in `[0, 1]`.
      scores (tf.Tensor): Confidence scores of shape `(N,)`.
      labels (tf.Tensor): Class indices of shape `(N,)`, dtype integer.
      pre_nms_k (Optional[int]): If provided, keep only the top-`k` by `scores`
        before NMS to reduce computation. Defaults to `500`.
      post_nms_k (Optional[int]): If provided, keep only the top-`k` by decayed
        scores after NMS. If `None`, keep all above the threshold. Defaults to `100`.
      score_threshold (float): Minimum decayed score to keep a mask. Defaults to `0.5`.
      sigma (float): Gaussian decay parameter used in Matrix NMS. Defaults to `0.5`.

    Returns:
      tf.Tensor: Indices (into the original `masks/scores/labels` tensors) of the
      kept detections after Matrix NMS, shape `(M,)`, dtype `int32`.

    Notes:
      * IoU is computed on binarized masks (`>= 0.5`).
      * Suppression is class-aware; cross-class pairs do not contribute to decay.
    """
    # Binarize masks at 0.5 threshold
    seg_masks = tf.cast(masks >= 0.5, dtype=tf.float32)  # shape: (N, H, W)
    mask_sum = tf.reduce_sum(seg_masks, axis=[1, 2])  # shape: (N,)

    # If desired, select top pre_nms_k by score to limit computation
    num_masks = tf.shape(scores)[0]
    if pre_nms_k is not None:
        num_selected = tf.minimum(pre_nms_k, num_masks)
    else:
        num_selected = num_masks
    topk_indices = tf.argsort(scores, direction='DESCENDING')[:num_selected]
    seg_masks = tf.gather(seg_masks, topk_indices)  # select masks by top scores
    labels_sel = tf.gather(labels, topk_indices)
    scores_sel = tf.gather(scores, topk_indices)
    mask_sum_sel = tf.gather(mask_sum, topk_indices)

    # Flatten masks for matrix operations
    N = tf.shape(seg_masks)[0]
    seg_masks_flat = tf.reshape(seg_masks, (N, -1))  # shape: (N, H*W)

    # Compute intersection and IoU matrix (N x N)
    intersection = tf.matmul(seg_masks_flat, seg_masks_flat, transpose_b=True)  # pairwise intersect counts
    # Expand mask areas to full matrices
    mask_sum_matrix = tf.tile(mask_sum_sel[tf.newaxis, :], [N, 1])  # shape: (N, N)
    union = mask_sum_matrix + tf.transpose(mask_sum_matrix) - intersection
    iou = intersection / (union + 1e-6)  # IoU matrix (avoid div-by-zero)
    # Zero out diagonal and lower triangle (keep i<j pairs)
    iou = tf.linalg.band_part(iou, 0, -1) - tf.linalg.band_part(iou, 0, 0)  # upper triangular without diagonal

    # Class-aware IoU: zero out IoU for pairs with different labels
    labels_matrix = tf.tile(labels_sel[tf.newaxis, :], [N, 1])  # each row is labels vector
    same_class = tf.cast(tf.equal(labels_matrix, tf.transpose(labels_matrix)), tf.float32)
    same_class = tf.linalg.band_part(same_class, 0, -1) - tf.linalg.band_part(same_class, 0, 0)
    decay_iou = iou * same_class  # IoU only for same-class pairs (upper tri)

    # Compute max IoU for each mask with any higher-scoring mask
    # (Since i<j is upper tri, for column j, relevant i are those with i < j)
    max_iou_per_col = tf.reduce_max(decay_iou, axis=0)
    comp_matrix = tf.tile(max_iou_per_col[..., tf.newaxis], [1, N])

    decay_matrix = tf.exp(-((decay_iou ** 2 - comp_matrix ** 2) / sigma))

    # Aggregate decay: for each column j, get the minimum decay factor across all i<j
    decay_coeff = tf.reduce_min(decay_matrix, axis=0)  # shape: (N,)
    decay_coeff = tf.where(tf.math.is_inf(decay_coeff), 1.0, decay_coeff)
    # (If no i<j, reduce_min gives +inf; replace inf with 1.0 meaning no suppression)

    # Decay the scores and filter by threshold
    new_scores = scores_sel * decay_coeff
    keep_mask = new_scores >= score_threshold                        # boolean mask of those above threshold
    new_scores = tf.where(keep_mask, new_scores, tf.zeros_like(new_scores))

    # Select top post_nms_k by the decayed scores
    if post_nms_k is not None:
        num_final = tf.minimum(post_nms_k, tf.shape(new_scores)[0])
    else:
        num_final = tf.shape(new_scores)[0]
    final_indices = tf.argsort(new_scores, direction='DESCENDING')[:num_final]
    final_indices = tf.boolean_mask(final_indices, tf.greater(tf.gather(new_scores, final_indices), 0))

    # Map back to original indices
    kept_indices = tf.gather(topk_indices, final_indices)
    return kept_indices

@tf.function
def get_instances(resized_h, resized_w, cate_preds, mask_preds, mask_feat, score_threshold):
    """
    Convert SOLO head outputs into per-instance masks/scores/classes.

    This function gathers all grid-cell/class pairs whose classification
    probability exceeds `score_threshold`, upsamples their corresponding mask
    logits to the given resized image shape, and returns tensors suitable for
    downstream NMS.

    Args:
        resized_h (int): Height of the resized/reference image for masks.
        resized_w (int): Width of the resized/reference image for masks.
        cate_preds (tf.Tensor): Category logits of shape `[1, sum(S_i*S_i), C]` where `C` includes background at index 0.
        mask_preds (tf.Tensor): Mask kernels of shape `[1, H_l, W_l, sum(S_i*S_i)]`.
        mask_feat (tf.Tensor): Mask features of shape `[1, H_l, W_l, E]`.
        score_threshold (float): Minimum class probability to keep a (cell, class) candidate.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        `(selected_masks, selected_scores, selected_classes)` where
            * `selected_masks` has shape `(K, resized_h, resized_w)` with values in `[0, 1]` (after sigmoid + bilinear upsampling).
            * `selected_scores` has shape `(K,)` with per-instance class probabilities (after softmax over classes, background removed).
            * `selected_classes` has shape `(K,)` with zero-based foreground class indices (i.e., background class 0 is excluded).

    Notes:
        * If no candidate passes the threshold, returns three empty tensors.
        * This function assumes background occupies class index `0` in `cate_preds` and removes it for thresholding.
    """
    # batch size
    B = tf.shape(mask_feat)[0]
    # featuremap spatial dims
    H, W = tf.shape(mask_feat)[1], tf.shape(mask_feat)[2]
    # number of kernels
    E = tf.shape(mask_feat)[3]

    cate_out = cate_preds[0]  # => shape [sum(S_i*S_i), num_classes]

    # Flatten mask_feat spatially: [B, H*W, E]
    mask_feat_flat = tf.reshape(mask_feat, [B, H * W, E])
    mask_out = tf.linalg.matmul(mask_feat_flat, mask_preds, transpose_b=True)
    mask_out = tf.reshape(mask_out, (B, H, W, tf.shape(mask_out)[2]))

    mask_out = mask_out[0]  # => shape (H_l, W_l, S_l^2)

    # Convert category logits => per-class probabilities (sigmoid)
    cate_prob = tf.nn.softmax(cate_out, axis=-1) # shape [sum(S_i*S_i), num_classes]

    # Convert mask logits => [0,1]
    mask_prob = tf.sigmoid(mask_out)  # shape (H_l, W_l, S_l^2)

    # Get rid of the background. Assume background has 0th index
    cate_prob = cate_prob[..., 1:]

    # Threshold in one shot (multi-label)
    # mask_bool[i, j] = True if cate_prob[i, j] >= threshold
    mask_bool = cate_prob >= tf.cast(score_threshold, tf.float32)  # shape (S_l*S_l, C)

    # Get indices of (cell, class) pairs and their scores
    idx = tf.where(mask_bool)

    # Early exit if nothing passes the threshold
    def _empty():
        return (tf.zeros([0, resized_h, resized_w], tf.float32),
                tf.zeros([0], tf.float32),
                tf.zeros([0], tf.int32))

    def _non_empty():
        selected_scores = tf.gather_nd(cate_prob, idx)  # [K]
        selected_classes = tf.cast(idx[:, 1], tf.int32)  # [K]
        selected_cells = tf.cast(idx[:, 0], tf.int32)  # [K]

        # Resize all S mask channels once (vectorized), then gather
        # mask_prob: [H_l, W_l, S] -> [K, H_l, W_l, 1] for tf.image.resize
        masks = tf.gather(mask_prob, selected_cells, axis=-1)  # [H_l, W_l, K]
        masks = tf.transpose(masks, [2, 0, 1])  # [K, H_l, W_l]
        masks = masks[..., tf.newaxis]  # [K, H_l, W_l, 1]

        # Bilinear resize with antialiasing (keeps values soft in [0,1])
        masks_up = tf.image.resize(
            masks, [resized_h, resized_w],
            method='bilinear', antialias=True
        )  # [K, H', W', 1]
        selected_masks = tf.squeeze(masks_up, axis=-1)  # [K, H', W']

        return selected_masks, selected_scores, selected_classes

    return tf.cond(tf.shape(idx)[0] > 0, _non_empty, _empty)


def postprocess_solo_outputs(
    cate_preds,          # Tensor of shape [1, sum(S_i*S_i), C]
    mask_preds,          # Tensor of shape [1, H_l, W_l, sum(S_i*S_i)]
    mask_feat,          # Tensor [1, H_l, W_l, E]
    resized_image_shape, # (resized_h, resized_w)
    score_threshold=0.5,
    nms_sigma=0.5
):
    """
    Convert raw SOLO outputs into final instance predictions.

    Runs candidate extraction (`get_instances`), applies Matrix NMS, and
    returns a sorted list of instance dictionaries compatible with downstream
    visualization or evaluation.

    Args:
        cate_preds (tf.Tensor): Category logits of shape `[1, sum(S_i*S_i), C]`.
        mask_preds (tf.Tensor): Mask kernels of shape `[1, H_l, W_l, sum(S_i*S_i)]`.
        mask_feat (tf.Tensor): Mask features of shape `[1, H_l, W_l, E]`.
        resized_image_shape (Tuple[int, int]): `(resized_h, resized_w)` used for upsampling masks.
        score_threshold (float, optional): Classification probability threshold for candidate selection. Defaults to `0.5`.
        nms_sigma (float, optional): Sigma parameter for Matrix NMS decay. Defaults to `0.5`.

    Returns:
        List[dict]: A list of instances. Each item has keys:
            * `"class_id"` (int): Zero-based class index (without background).
            * `"score"` (float): Final post-NMS score.
            * `"mask"` (np.ndarray): Binary mask of shape
            `(resized_h, resized_w)`, `dtype=uint8` with values in `{0,1}`.
        The list is sorted by descending `score`.

    Notes:
        * If no candidates remain after thresholding/NMS, an empty list is returned.
        * Returned masks are binarized at `0.5` after sigmoid.
    """
    resized_h, resized_w = resized_image_shape

    all_masks_arr_tf, all_scores_arr_tf, all_classes_arr_tf = get_instances(resized_h, resized_w, cate_preds, mask_preds, mask_feat, score_threshold)

    # If no candidates, return empty
    if all_masks_arr_tf.shape[0] == 0:
        return []

    # Matrix NMS
    kept_indices = matrix_nms(
        masks=all_masks_arr_tf,
        scores=all_scores_arr_tf,
        labels=all_classes_arr_tf,
        post_nms_k=None,
        score_threshold=score_threshold,
        sigma=nms_sigma
    )

    final_masks = tf.gather(all_masks_arr_tf, kept_indices, axis=0).numpy()
    final_scores = tf.gather(all_scores_arr_tf, kept_indices, axis=0).numpy()
    final_classes = tf.gather(all_classes_arr_tf, kept_indices, axis=0).numpy()
    final_masks = (final_masks > 0.5).astype(np.uint8)

    # Build the final instance list
    instances = []
    for i in range(len(kept_indices.numpy())):
        instances.append({
            # +1 if needed, depends on your label indexing
            "class_id": int(final_classes[i]),
            "score": float(final_scores[i]),
            "mask": final_masks[i]
        })

    # Sort again by descending final score for convenience
    instances = sorted(instances, key=lambda x: x["score"], reverse=True)
    return instances


def draw_solo_masks(
    original_image: np.ndarray,
    instances: list,
    show_labels=True,
    class_names=None
):
    """
    Overlay instance masks and optional labels on an image.

    Applies semi-transparent color overlays for each instance mask and, if
    requested, draws a class label near the first pixel of the mask.

    Args:
        original_image (np.ndarray): Input image on which to draw. Typically **BGR** (OpenCV convention), `dtype=uint8`, shape `(H, W, 3)`.
        instances (List[dict]): List of instance dicts as returned by `postprocess_solo_outputs` with keys `class_id`, `score`, and `mask` (binary `(h, w)` array).
        show_labels (bool, optional): Whether to render text labels. Defaults to `True`.
        class_names (Mapping[int, str] | Sequence[str] | None, optional): Mapping from class id to readable name. If provided and indexable by `class_id`, the corresponding name is used for the label.

    Returns:
        np.ndarray: Annotated image (same shape and dtype as `original_image`).

    Notes:
        * Overlay color is randomly sampled for each instance.
        * If `class_names` is missing a key, a fallback `"ID=<id>, <score>"` label is used.
    """
    vis_image = original_image.copy()
    orig_h, orig_w = vis_image.shape[:2]

    for inst in instances:
        mask_resized = inst["mask"]  # (resized_h, resized_w) => {0,1}
        # Upsample to original image size
        mask_orig = cv2.resize(
            mask_resized.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        # mask_orig => shape (orig_h, orig_w) in {0,1}

        # Color overlay
        color = [random.randint(0, 255) for _ in range(3)]
        alpha = 0.5
        vis_image[mask_orig == 1] = (
            alpha * np.array(color) + (1 - alpha) * vis_image[mask_orig == 1]
        )

        # Draw class label near the first pixel of mask
        ys, xs = np.where(mask_orig == 1)
        if len(ys) > 0:
            y0, x0 = ys[0], xs[0]
            score_str = f"{inst['score']:.2f}"
            if class_names and (0 <= inst['class_id'] < len(class_names)):
                label_str = f"{class_names.get(inst['class_id'])}"#: {score_str}"
            else:
                label_str = f"ID={inst['class_id']}, {score_str}"
            cv2.putText(
                vis_image, label_str if show_labels == True else "",
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

    return vis_image

# Enable dynamic memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    cfg = DynamicSOLOConfig()

    classes_path = cfg.classes_path
    model_path = cfg.test_model_path
    input_shape = (cfg.img_height, cfg.img_width, 3)
    coco = COCO(cfg.train_annotation_path)
    categories = coco.loadCats(coco.getCatIds())
    # Create dictionary: {category_id: category_name}
    coco_classes = {cat['id']: cat['name'] for cat in categories}

    # Workaround because of NGC's TensorFlow version
    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    img_height, img_width = cfg.img_height, cfg.img_width
    solo = SOLOModel(
        input_shape=(img_height, img_width, 3),  # Example shape
        num_classes=num_classes,
        num_stacked_convs=7,
        head_input_channels=256,
        mask_kernel_channels=256,
        grid_sizes=cfg.grid_sizes
    )
    solo.build((None, img_height, img_width, 3))
    solo.load_weights(model_path)

    if not os.path.exists('images/res/'): os.mkdir('images/res/')
    path_dir = os.listdir('images/test')

    for k, filename in enumerate(path_dir):
        # --- Preprocess ---
        image_path = f'images/test/{filename}'
        img_resized, (orig_h, orig_w), img_rgb = preprocess_image(image_path, input_size=input_shape[:2])
        # img_resized:  resized to (input_size)
        # (orig_h, orig_w): original image dims
        # img_rgb: original image in RGB

        # Expand dims for batch
        img_batch = np.expand_dims(img_resized, axis=0)
        img_batch = tf.convert_to_tensor(img_batch)

        class_outputs, mask_outputs, mask_feat = solo.predict(img_batch)

        # Postprocess All Levels (scales)
        all_instances = []
        instances = postprocess_solo_outputs(
            cate_preds=class_outputs,
            mask_preds=mask_outputs,
            mask_feat=mask_feat,
            resized_image_shape=input_shape[:2],
            score_threshold=cfg.score_threshold,
            nms_sigma=0.5
        )

        # Draw an image for saving
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        annotated_img = draw_solo_masks(img_bgr, instances, show_labels=True, class_names=coco_classes)

        # Save output
        output_filename = os.path.basename(image_path)
        save_path = os.path.join('images/res', output_filename)
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved annotated image: {save_path}")

    exit(0)


