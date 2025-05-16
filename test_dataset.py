"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script draws dataset with masks by categories to understand what data we fit to the model.
"""


import cv2
import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import DynamicSOLOConfig
from coco_dataset import create_coco_tf_dataset, get_classes
import shutil


def save_masks_by_category(
        image_tensor,
        ms_targets,
        category_names,
        img_num,
        out_dir="masks_output"
):
    """
    For each scale in `ms_targets`, separate the mask channels by category and
    save the results to disk in a folder structure like:
        masks_output/
          catName_0/
            scale0_channel0.png
            scale0_channel5.png
            ...
          catName_1/
            scale1_channel3.png
            ...

    Args:
        image_tensor (tf.Tensor): shape=(H, W, 3), in [0,1].
        ms_targets (dict): e.g. contains "cate_target_0", "mask_target_0", etc.
        category_names(dict):
                       a dict mapping category_id -> readable category name.
        out_dir (str): root directory to save mask images.
    """
    # Make sure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Convert image to Numpy
    image_np = image_tensor.numpy()  # shape=(H, W, 3)
    img_h, img_w = image_np.shape[:2]

    # Figure out the scale indices from ms_targets
    scale_indices = []
    for key in ms_targets.keys():
        if key.startswith("mask_target_"):
            idx = int(key.split("_")[-1])
            scale_indices.append(idx)
    scale_indices.sort()

    # Loop over scales
    for scale_idx in scale_indices:
        cate_key = f"cate_target_{scale_idx}"
        mask_key = f"mask_target_{scale_idx}"

        cate_tensor = ms_targets[cate_key]  # shape=(grid_size, grid_size)
        mask_tensor = ms_targets[mask_key]  # shape=(feat_h, feat_w, grid_size^2)

        cate_np = cate_tensor[0].numpy()
        mask_np = mask_tensor[0].numpy()

        feat_h, feat_w, num_channels = mask_np.shape
        grid_size = cate_np.shape[0]  # e.g. 40, 36, etc.

        # Sanity check: grid_size^2 should match num_channels
        assert grid_size * grid_size == num_channels, \
            f"Mismatch: grid_size^2={grid_size * grid_size} vs mask channels={num_channels}"

        # Iterate over each channel in mask_target
        for ch in range(num_channels):
            # Derive the grid cell (gy, gx) from channel index
            gy = ch // grid_size
            gx = ch % grid_size

            cat_id = cate_np[gy, gx]
            if cat_id == -1:
                # No object in this cell
                continue

            # Get the mask for this channel
            mask_ch = mask_np[..., ch]  # shape=(feat_h, feat_w)

            # If the mask is all zeros, skip to avoid extra blank images
            if np.max(mask_ch) == 0:
                continue

            # Resize mask to match the original (H, W)
            # We use INTER_NEAREST to preserve the hard edges
            mask_resized = cv2.resize(
                mask_ch.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Prepare an overlay or just the masked region
            # Here we show an example overlay in red
            overlay_color = (1.0, 0.0, 0.0)  # Red in [0,1]
            overlay = image_np.copy()

            # Make a colored overlay wherever mask==1
            # If you prefer pure mask, skip the overlay logic
            overlay[mask_resized > 0] = overlay_color

            # Convert overlay to 8-bit
            overlay_8u = (overlay * 255).astype(np.uint8)

            # Create category folder
            # If you have a dict or list of category_names, do something like:
            cat_name = category_names.get(cat_id) if cat_id < len(category_names) else f"cat_{cat_id}"
            cat_dir = os.path.join(out_dir, cat_name)
            os.makedirs(cat_dir, exist_ok=True)

            # Construct a filename
            # Example: scale0_channel12.png
            out_filename = f"img_{img_num}_scale{scale_idx}_channel{ch}.png"
            out_path = os.path.join(cat_dir, out_filename)

            # Save to disk
            cv2.imwrite(out_path, cv2.cvtColor(overlay_8u, cv2.COLOR_RGB2BGR))

    print(f"Masks saved under '{out_dir}' separated by category.")

if __name__ == '__main__':
    cfg = DynamicSOLOConfig()
    coco = COCO(cfg.train_annotation_path)
    categories = coco.loadCats(coco.getCatIds())
    # Create dictionary: {category_id: category_name}
    coco_classes = {cat['id']: cat['name'] for cat in categories}

    num_classes = len(coco_classes)
    img_height, img_width = cfg.img_height, cfg.img_width

    ds = create_coco_tf_dataset(
        coco_annotation_file=cfg.train_annotation_path,
        coco_img_dir=cfg.images_path,
        num_classes=num_classes,
        target_size=(img_height, img_width),
        batch_size=1,
        grid_sizes=cfg.grid_sizes,
        scale=cfg.image_scales[0],
        augment=False,
        number_images=20
    )

    out_dir = 'images/dataset_test'
    shutil.rmtree(out_dir, ignore_errors=True)

    for img_num, (image_tensor, ms_targets) in enumerate(ds):
        save_masks_by_category(image_tensor[0], ms_targets, category_names=coco_classes, img_num=img_num, out_dir=out_dir)


