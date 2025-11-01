"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains classes and functions COCO dataset generation.
"""


import os
import math
import numpy as np
import random
import tensorflow as tf
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


def get_classes(classes_path):
    """
    Loads class names from a text file.

    Each line in the file is assumed to represent a single class name.
    The function reads all lines, strips whitespace, and returns a list of class names.

    Args:
        classes_path (str): Path to the text file containing class names, one per line.

    Returns:
        List[str]: A list of class names as strings, with leading and trailing whitespace removed.
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def ann_to_mask(ann, height, width):
    """
    Converts a COCO annotation into a binary mask (0 or 1). Supports both polygon and RLE formats.

    Args:
        ann (dict): COCO-style annotation dictionary containing a 'segmentation' field.
            Can be a list of polygons or an RLE (Run-Length Encoding) dict.
        height (int): Height of the output mask.
        width (int): Width of the output mask.

    Returns:
        np.ndarray: A binary mask of shape (height, width), where 1 indicates the object and 0 the background.

    Raises:
        TypeError: If the segmentation format is not a list or dict.
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = coco_mask.frPyObjects(segm, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segm, dict):
        # RLE
        rle = segm
    else:
        raise TypeError("Unknown segmentation format.")

    mask = coco_mask.decode(rle)
    # If multiple segments => combine
    if len(mask.shape) == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    return mask


def compute_scale_ranges(image_height, image_width, num_levels=4, reduction_factor=2):
    """
    Computes object scale ranges for different feature levels based on image size.

    This function partitions the square root of the image area into `num_levels` ranges,
    optionally reduced by a `reduction_factor`. These scale ranges can be used to assign
    objects to appropriate levels in a feature pyramid.

    Args:
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
        num_levels (int, optional): Number of feature levels. Defaults to 4.
        reduction_factor (float, optional): Factor to reduce bin width, increasing sensitivity to small objects. Defaults to 2.

    Returns:
        List[Tuple[float, float]]: A list of tuples, where each tuple represents a (lower, upper) bound of the scale range
        for a corresponding level. The last range has an upper bound of infinity.
    """
    max_sqrt_area = math.sqrt(image_height * image_width)
    bin_width = max_sqrt_area / (num_levels * reduction_factor)

    scale_ranges = []
    lower = 0.0
    for level_idx in range(num_levels):
        upper = (level_idx + 1) * bin_width

        # for the last level, set upper to a large sentinel
        if level_idx == num_levels - 1:
            upper = float('inf')

        scale_ranges.append((lower, upper))
        lower = (level_idx + 1) * bin_width

    return scale_ranges


def generate_solo_targets_single_scale(
    anns,
    resized_masks,
    resized_bboxes,
    img_shape,       # (img_h, img_w): original image shape
    grid_size,
    scale_range,     # (scale_min, scale_max) for sqrt(area), e.g. (0, 64)
    scale=0.25,       # scale factor for feature map size (default 1/4)
    center_radius=1  # radius (in grid cells) around center to assign
):
    """
    Creates category & mask targets for a single SOLO grid scale,
    with:
      1) Scale filtering: only assign objects with sqrt(area) in scale_range
      2) Center region assignment: each object is assigned to a neighborhood
         of cells around its center of mass.

    Args:
        anns (list): Each item is a dict with:
            - 'category_id' (int): COCO-style class ID
        resized_masks (list): list of np.ndarray. shape=(img_h, img_w), masks in original image space
        resized_bboxes (list): list of bounding boxes. Each element is a box [x, y, w, h] in (img_h, img_w) coords
        img_shape (tuple): (img_h, img_w) for original image
        grid_size (int): number of cells along each spatial axis of SOLO grid
        scale_range (tuple): (scale_min, scale_max),
            e.g. (0, 64) => handle objects with sqrt(area) in [0,64)
        scale (float): scale factor to compute feature map size, default=1/4
        center_radius (int): how many grid cells around the center to assign

    Returns:
        cate_target_tf (tf.Tensor): shape=[grid_size, grid_size] or [S_i, S_i],
                                 storing class IDs or -1 if no object
        mask_target_tf (tf.Tensor): shape=[feat_shape, feat_shape, grid_size^2] or [H, W, S_i*S_i],
                                 storing binary masks for each cell/channel.
    """
    img_h, img_w = img_shape
    scale_min, scale_max = scale_range  # e.g. (0, 64)

    # Compute feature map shape from scale
    feat_h = int(round(img_h * scale))
    feat_w = int(round(img_w * scale))

    # Prepare empty targets
    cate_target = np.full((grid_size, grid_size), -1, dtype=np.int32)
    mask_channels = grid_size * grid_size
    mask_target = np.zeros((feat_h, feat_w, mask_channels), dtype=np.uint8)

    # Compute scale factors to map from original image space -> feature-map space
    height_scale = feat_h / float(img_h)
    width_scale  = feat_w / float(img_w)

    # Compute size of each cell in feature-map coordinates
    cell_h = feat_h / float(grid_size)
    cell_w = feat_w / float(grid_size)

    # Loop over annotations
    for idx, ann in enumerate(anns):
        cat_id = ann['category_id']
        seg_mask_original = resized_masks[idx]  # (img_h, img_w)
        x, y, w, h = resized_bboxes[idx]             # bounding box in original image coords

        # Filter out objects that do not fit this scale range
        #     We'll use sqrt(area) of the bounding box for scale filtering.
        area = w * h
        if area <= 0:
            continue
        obj_size = math.sqrt(area)
        if (obj_size < scale_min) or (obj_size >= scale_max):
            # This object doesn't belong to this scale
            continue

        # Get pixel coordinates of the mask to compute center of mass
        coords = np.argwhere(seg_mask_original > 0)
        if len(coords) == 0:
            # Empty mask => skip
            continue
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        # Center of mass in original image coords
        y_com = np.mean(y_coords)
        x_com = np.mean(x_coords)

        # Convert center to feature-map coords
        x_com_fpn = x_com * width_scale
        y_com_fpn = y_com * height_scale

        # Compute the center cell in the grid
        gx_center = int(np.floor(x_com_fpn / cell_w))
        gy_center = int(np.floor(y_com_fpn / cell_h))
        gx_center = max(min(gx_center, grid_size - 1), 0)
        gy_center = max(min(gy_center, grid_size - 1), 0)

        # Resize the mask to the feature map resolution
        seg_mask_fpn = cv2.resize(
            seg_mask_original,
            (feat_w, feat_h),
            interpolation=cv2.INTER_NEAREST
        )

        # "Center region" assignment: assign to cells within center_radius
        for dy in range(-center_radius, center_radius + 1):
            for dx in range(-center_radius, center_radius + 1):
                gx = gx_center + dx
                gy = gy_center + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    channel_idx = gy * grid_size + gx

                    # Assign only if cell is not taken
                    if cate_target[gy, gx] == -1:
                        cate_target[gy, gx] = cat_id
                        mask_target[..., channel_idx] = seg_mask_fpn

    # Convert to TF tensors
    cate_target_tf = tf.convert_to_tensor(cate_target, dtype=tf.int32)
    mask_target_tf = tf.convert_to_tensor(mask_target, dtype=tf.uint8)

    return cate_target_tf, mask_target_tf


def generate_solo_targets_multi_scale(
        anns, resized_masks, resized_bboxes, img_shape, grid_sizes, scale, scale_ranges
):
    """
    Generates classification and mask targets for multiple feature map scales.

    For each scale in `grid_sizes`, this function calls `generate_solo_targets_single_scale`
    to produce category and mask targets. The results from all scales are then concatenated.

    Args:
        anns (list): Each item is a dict with:
            - 'category_id' (int): COCO-style class ID
        resized_masks (list): list of np.ndarray. shape=(img_h, img_w), masks in original image space
        resized_bboxes (list): list of bounding boxes. Each element is a box [x, y, w, h] in (img_h, img_w) coords
        img_shape (Tuple[int, int]): Shape of the image as (height, width).
        grid_sizes (List[int]): List of grid sizes corresponding to different feature map levels.
        scale (float): Downsampling scale factor from the image to the feature map.
        scale_ranges (List[Tuple[float, float]]): List of (lower, upper) scale bounds for each grid level.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - cate_targets (tf.Tensor): Concatenated classification targets of shape [sum(S_i * S_i),].
            - mask_targets (tf.Tensor): Concatenated mask targets of shape [H, W, sum(S_i * S_i)].
    """

    cate_targets = []
    mask_targets = []

    for i, gs in enumerate(grid_sizes):
        cate_t, mask_t = generate_solo_targets_single_scale(
            anns=anns,
            resized_masks=resized_masks,
            resized_bboxes=resized_bboxes,
            img_shape=img_shape,
            scale=scale,
            grid_size=gs,
            scale_range=scale_ranges[i],
            center_radius=0
        )
        cate_targets.append(tf.reshape(cate_t, [tf.shape(cate_t)[0] * tf.shape(cate_t)[1]]))
        mask_targets.append(mask_t)

    cate_targets = tf.concat(cate_targets, axis=0)  # [sum(S_i*S_i),]
    mask_targets = tf.concat(mask_targets, axis=2)  # [H, W, sum(S_i*S_i)]
    return cate_targets, mask_targets


class CocoGenerator:
    """
    A data generator for COCO-formatted datasets.

    This class loads images and annotations from a COCO dataset, resizes images and masks,
    and generates multi-scale SOLO targets for training.

    Attributes:
        coco (COCO): An instance of the COCO API containing annotations.
        coco_img_dir (str): Path to the directory containing COCO images.
        grid_sizes (List[int]): List of grid sizes for different feature levels.
        scale (float): Downsampling factor from image resolution to feature map resolution. Based on P2 FPN level
        num_classes (int): Number of object categories.
        target_size (Tuple[int, int]): Size (height, width) to resize images and masks.
        shuffle (bool): Whether to shuffle the image IDs.
        augment (bool): Whether to apply data augmentation.
        number_images (int): Number of images per generation. Use all images if the parameter set to None
    """
    def __init__(self, coco, coco_img_dir,
        grid_sizes=[40, 36, 24, 16],
        scale=0.25,
        target_size=(320, 320),
        shuffle=True,
        augment=True,
        number_images=None):
        self._coco = coco
        self._image_ids = list(coco.imgs.keys())
        self._coco_img_dir = coco_img_dir
        self._grid_sizes = grid_sizes
        self._scale = scale
        self._target_size = target_size
        self._shuffle = shuffle
        self._augment = augment
        self._number_images = number_images
        self._scale_ranges = compute_scale_ranges(target_size[0], target_size[1], num_levels=len(grid_sizes))

    def generate(self):
        """
        Preprocesses images and generates their corresponding targets.

        The method performs the following steps:
            1. Loads an image and its annotations from the COCO dataset.
            2. Resizes the image to `target_size`.
            3. Builds resized binary masks for each annotation.
            4. Generates classification and mask targets for multiple scales.

        Yields:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                - image_tensor (tf.Tensor): Resized image as a float32 tensor of shape [H, W, 3], with pixel values in [0, 1].
                - cate_targets (tf.Tensor): Concatenated classification targets of shape [sum(S_i * S_i),].
                - mask_targets (tf.Tensor): Concatenated mask targets of shape [H, W, sum(S_i * S_i)].
        """
        if self._shuffle:
            random.shuffle(self._image_ids)

        if self._number_images is not None:
            considered_image_ids = random.sample(self._image_ids, self._number_images)
        else:
            considered_image_ids = self._image_ids

        for img_id in considered_image_ids:
            # Load image
            img_info = self._coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            path = os.path.join(self._coco_img_dir, file_name)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

            # Annotations
            ann_ids = self._coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = self._coco.loadAnns(ann_ids).copy()

            # Convert masks and bounding boxes before augmentation
            masks = []
            for ann in anns:
                masks.append(ann_to_mask(ann, orig_h, orig_w))

            if not masks:
                continue
            masks = np.stack(masks, axis=0)

            # Apply augmentations before resizing
            if self._augment:
                image, anns, masks = self._apply_augmentations(image, anns, masks)

            # Resize image
            new_h, new_w = self._target_size
            image = cv2.resize(image, (new_w, new_h))
            image_tensor = image.astype(np.float32) / 255.0

            scale_y = new_h / float(orig_h)
            scale_x = new_w / float(orig_w)

            resized_masks = []
            resized_bboxes = []
            # For each annotation -> resized mask & bounding box
            for idx, ann in enumerate(anns):
                seg_mask_resized = cv2.resize(masks[idx], (new_w, new_h))
                seg_mask_resized = (seg_mask_resized > 0.5).astype(np.uint8)
                resized_masks.append(seg_mask_resized)

                x, y, w, h = ann['bbox']
                resized_bboxes.append([
                    x * scale_x,
                    y * scale_y,
                    w * scale_x,
                    h * scale_y
                ])


            # Generate multi-scale SOLO targets
            cate_targets, mask_targets = generate_solo_targets_multi_scale(
                anns,
                resized_masks,
                resized_bboxes,
                img_shape=self._target_size,
                grid_sizes=self._grid_sizes,
                scale=self._scale,
                scale_ranges=self._scale_ranges
            )

            image_tensor = tf.convert_to_tensor(image_tensor)

            # Yield
            yield image_tensor, cate_targets, mask_targets

    def _apply_augmentations(self, image, anns, masks):
        """
        Applies random data augmentations to the input image, annotations, and masks.

        The following augmentations are applied with 50% probability each:
            - Horizontal flip (image, masks, and bounding boxes are mirrored horizontally).
            - Brightness adjustment (randomly lighten or darken the image by up to 20%).
            - Random scaling (rescale image, masks, and bounding boxes by a factor between 0.8 and 1.2).
            - Random cropping (crop a random region, discarding no more than 20% per side, adjusting bounding boxes accordingly).

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, 3).
            anns (List[dict]): List of annotation dictionaries, each containing a 'bbox' field in COCO format [x, y, width, height].
            masks (np.ndarray): Binary masks as a NumPy array of shape (N, H, W), where N is the number of instances.

        Returns:
            Tuple[np.ndarray, List[dict], np.ndarray]:
                - The augmented image as a NumPy array.
                - The updated list of annotations with adjusted bounding boxes.
                - The augmented binary masks as a NumPy array.

        Notes:
            - Annotations for objects that no longer have valid bounding boxes after cropping (width or height <= 1) are discarded.
        """
        h, w = image.shape[:2]

        # Random horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            masks = np.flip(masks, axis=2)
            for ann in anns:
                x, y, box_w, box_h = ann['bbox']
                ann['bbox'] = [w - x - box_w, y, box_w, box_h]

        # Random brightness
        if random.random() < 0.5:
            factor = 1.0 + (random.random() - 0.5) * 0.4  # +/-20%
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Random scaling
        if random.random() < 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
            masks = np.array([cv2.resize(mask, (int(w * scale_factor), int(h * scale_factor))) for mask in masks])
            for ann in anns:
                x, y, box_w, box_h = ann['bbox']
                ann['bbox'] = [x * scale_factor, y * scale_factor, box_w * scale_factor, box_h * scale_factor]
            h, w = image.shape[:2]

        # Random crop (discard no more than 20% per side)
        if random.random() < 0.5:
            max_crop_x = int(w * 0.2)
            max_crop_y = int(h * 0.2)

            crop_x1 = random.randint(0, max_crop_x)
            crop_y1 = random.randint(0, max_crop_y)
            crop_x2 = random.randint(w - max_crop_x, w)
            crop_y2 = random.randint(h - max_crop_y, h)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            masks = masks[:, crop_y1:crop_y2, crop_x1:crop_x2]
            new_anns = []
            for idx, ann in enumerate(anns):
                x, y, box_w, box_h = ann['bbox']
                new_x = max(0, x - crop_x1)
                new_y = max(0, y - crop_y1)
                new_w = max(0, min(box_w, crop_w - new_x))
                new_h = max(0, min(box_h, crop_h - new_y))
                if new_w > 1 and new_h > 1:
                    ann['bbox'] = [new_x, new_y, new_w, new_h]
                    new_anns.append(ann)
            anns = new_anns

        return image, anns, masks


def create_coco_tf_dataset(
        coco_annotation_file, coco_img_dir,
        grid_sizes=[40, 36, 24, 16],
        scale=0.25,
        target_size=(320, 320),
        batch_size=2,
        shuffle=True,
        augment=True,
        number_images=None
):
    """
    Creates a TensorFlow dataset for training SOLO-based instance segmentation models using COCO annotations.

    This function initializes a `CocoGenerator` to load images and annotations,
    applies optional augmentations, and formats the output into a `tf.data.Dataset`
    with multi-scale SOLO targets.

    Args:
        coco_annotation_file (str): Path to the COCO annotation JSON file.
        coco_img_dir (str): Directory containing the corresponding COCO images.
        grid_sizes (List[int], optional): List of grid sizes corresponding to different feature levels. Defaults to [40, 36, 24, 16].
        scale (float, optional): Downsampling scale factor from image to feature map resolution. Defaults to 0.25. Based on P2 FPN level
        target_size (Tuple[int, int], optional): Desired (height, width) to resize all images and masks to. Defaults to (320, 320).
        batch_size (int, optional): Number of samples per batch. Defaults to 2.
        shuffle (bool, optional): Whether to shuffle the dataset before generating batches. Defaults to True.
        augment (bool, optional): Whether to apply random data augmentations. Defaults to True.
        number_images (int): Number of images per generation. Use all images if the parameter set to None

    Returns:
        tf.data.Dataset: A dataset yielding batched tuples
            (images, cate_target, mask_target) with dtypes and shapes:

            - images: tf.float32, shape [batch_size, H, W, 3],
              where H, W = target_size.
            - cate_target: tf.int32, shape [batch_size, sum(S_i * S_i)].
            - mask_target: tf.uint8, shape [batch_size, H, W, sum(S_i * S_i)].
              where S_i are the grid sizes in `grid_sizes`.
    """
    coco = COCO(coco_annotation_file)

    coco_generator = CocoGenerator(
            coco, coco_img_dir,
            grid_sizes=grid_sizes,
            scale=scale,
            target_size=target_size,
            shuffle=shuffle,
            augment=augment,
            number_images=number_images
        )

    output_signature = (
        tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32, name="images"),
        tf.TensorSpec(shape=(sum(x**2 for x in grid_sizes),), dtype=tf.int32, name="cate_target"),
        tf.TensorSpec(shape=(int(round(target_size[0] * scale)), int(round(target_size[1] * scale)), sum(x**2 for x in grid_sizes)),
                      dtype=tf.uint8, name="mask_target"),
    )

    dataset = tf.data.Dataset.from_generator(
        coco_generator.generate,
        output_signature=output_signature
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset