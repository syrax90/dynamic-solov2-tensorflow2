"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script draws dataset with masks by categories to understand what data we fit to the model.
"""


import os
from pycocotools.coco import COCO
import tensorflow as tf
import numpy as np
import cv2
import random

from config import DynamicSOLOConfig
from coco_dataset import create_coco_tf_dataset
from coco_dataset_optimized import create_coco_tfrecord_dataset
import shutil

def draw_solo_instances(
    image,                 # tf.Tensor or np.ndarray, HxWx3, RGB, [0,1] or [0,255]
    cate_target,           # tf.Tensor or np.ndarray, shape [sum(S_i^2)], int32, -1 for empty else category_id
    mask_target,           # tf.Tensor or np.ndarray, shape [Hf, Wf, sum(S_i^2)], uint8 {0,1}
    class_names=None,      # dict {category_id: "name"} (same role as in your snippet)
    show_labels=True       # follow your snippet's switch
):
    """
    Returns a BGR uint8 visualization with colored masks and optional class labels.
    Coloring, transparency and blending follow the behavior in draw_solo_masks():
      - random color per instance
      - alpha = 0.5
      - vis[mask==1] = alpha*color + (1-alpha)*vis[mask==1]
      - label near the first pixel of the mask
    """

    # --- to numpy ---
    if isinstance(image, tf.Tensor):       image = image.numpy()
    if isinstance(cate_target, tf.Tensor): cate_target = cate_target.numpy()
    if isinstance(mask_target, tf.Tensor): mask_target = mask_target.numpy()

    # --- image to uint8 BGR (so cv2.imwrite works as expected) ---
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    H, W = vis.shape[:2]
    Hf, Wf, C = mask_target.shape
    assert C == cate_target.shape[0], "mask_target channels and cate_target length must match"

    # Upsample all masks to image size using NEAREST (binary kept)
    if (Hf != H) or (Wf != W):
        up_masks = np.zeros((H, W, C), dtype=np.uint8)
        for c in range(C):
            up_masks[..., c] = cv2.resize(mask_target[..., c], (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        up_masks = mask_target

    # positive channels (instances)
    pos_idx = np.where(cate_target >= 0)[0]
    if pos_idx.size == 0:
        return vis

    alpha = 0.5

    for ch in pos_idx:
        class_id = int(cate_target[ch])
        mask_bin = (up_masks[..., ch] > 0)

        if mask_bin.sum() == 0:
            continue

        # Random color per instance (same principle as your snippet)
        color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)

        # Per-pixel alpha blend exactly like:
        # vis[mask==1] = alpha*color + (1-alpha)*vis[mask==1]
        m = mask_bin
        if m.any():
            region = vis[m].astype(np.float32)
            vis[m] = (alpha * color + (1.0 - alpha) * region).astype(np.uint8)

        # Label near first pixel in the mask (if enabled)
        if show_labels:
            ys, xs = np.where(m)
            if len(ys) > 0:
                y0, x0 = int(ys[0]), int(xs[0])
                if isinstance(class_names, dict) and (class_id in class_names):
                    label_str = f"{class_names.get(class_id)}"
                else:
                    label_str = f"ID={class_id}"
                cv2.putText(
                    vis, label_str,
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

    return vis

def save_dataset_preview(dataset, coco_classes, out_dir, max_images=50, show_labels=True):
    """
    Save a grid-free preview of SOLO targets for a dataset.

    Iterates over a `tf.data.Dataset` that yields `(images, cate_targets, mask_targets)`
    batches, renders each sample with `draw_solo_instances`, and writes PNG files to
    `out_dir` until `max_images` previews are saved.

    Args:
      dataset: `tf.data.Dataset` where each batch is a tuple:
        - images: Tensor of shape (B, H, W, 3), RGB, float in [0,1] or uint8.
        - cate_targets: Tensor of shape (B, C), int32; -1 for empty, category_id otherwise.
        - mask_targets: Tensor of shape (B, Hf, Wf, C), uint8 {0,1}.
      coco_classes: `dict[int, str]` mapping category_id → class name for labeling.
      out_dir: Destination directory to store rendered PNG files.
      max_images: Maximum number of samples to save across all batches.
      show_labels: If True, overlay class labels on the previews.

    Returns:
      None
    """
    import os, cv2
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for batch in dataset:
        images, cate_targets, mask_targets = batch
        bs = images.shape[0]
        for b in range(bs):
            vis = draw_solo_instances(
                images[b].numpy(),
                cate_targets[b].numpy(),
                mask_targets[b].numpy(),
                class_names=coco_classes,
                show_labels=show_labels
            )
            cv2.imwrite(os.path.join(out_dir, f"sample_{saved:04d}.png"), vis)
            saved += 1
            if saved >= max_images:
                return

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    cfg = DynamicSOLOConfig()
    coco = COCO(cfg.train_annotation_path)
    categories = coco.loadCats(coco.getCatIds())
    # Create dictionary: {category_id: category_name}
    coco_classes = {cat['id']: cat['name'] for cat in categories}

    num_classes = len(coco_classes)
    img_height, img_width = cfg.img_height, cfg.img_width

    if cfg.use_optimized_dataset:
        ds = create_coco_tfrecord_dataset(
            train_tfrecord_directory=cfg.tfrecord_dataset_directory_path,
            target_size=(img_height, img_width),
            batch_size=cfg.batch_size,
            grid_sizes=cfg.grid_sizes,
            scale=cfg.image_scales[0],
            augment=cfg.augment,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            number_images=cfg.number_images)
    else:
        ds = create_coco_tf_dataset(
            coco_annotation_file=cfg.train_annotation_path,
            coco_img_dir=cfg.images_path,
            target_size=(img_height, img_width),
            batch_size=cfg.batch_size,
            grid_sizes=cfg.grid_sizes,
            scale=cfg.image_scales[0],
            augment=False,
            number_images=cfg.number_images
        )

    out_dir = 'images/dataset_test'
    shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)
    save_dataset_preview(ds, coco_classes, out_dir, max_images=200)  # adjust as needed
    print(f"Saved previews to: {out_dir}")


