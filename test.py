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
import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def preprocess_image(image_path, input_size=(320, 320)):
    """
    Loads an image from disk, resizes it to a fixed input size, and normalizes pixel values to [0, 1].

    The function also returns the original image shape and the unnormalized RGB image

    Args:
        image_path (str): Path to the input image file.
        input_size (Tuple[int, int], optional): Target size (width, height) to resize the image to. Defaults to (320, 320).

    Returns:
        Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
            - img_resized (np.ndarray): The resized and normalized image of shape (H, W, 3), dtype float32.
            - original_shape (Tuple[int, int]): The original image shape as (height, width).
            - img (np.ndarray): The original RGB image before resizing and normalization.

    Raises:
        ValueError: If the image cannot be read from the given path.
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
def matrix_nms(masks, scores, labels, pre_nms_k=500, post_nms_k=100, score_threshold=0.05, sigma=0.5):
    """
    Perform class-wise Matrix NMS on instance masks.

    Parameters:
        masks (tf.Tensor): Tensor of shape (N, H, W) with each mask as a sigmoid probability map (0~1).
        scores (tf.Tensor): Tensor of shape (N,) with confidence scores for each mask.
        labels (tf.Tensor): Tensor of shape (N,) with class labels for each mask (ints).
        pre_nms_k (int): Number of top-scoring masks to keep before applying NMS.
        post_nms_k (int): Number of final masks to keep after NMS.
        score_threshold (float): Score threshold to filter out masks after NMS (default 0.5).
        sigma (float): Sigma value for Gaussian decay.

    Returns:
        tf.Tensor: Tensor of indices of masks kept after suppression.
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


def postprocess_solo_outputs(
    cate_preds,          # list[Tensor], each (1, S_l, S_l, C) => per-class logits
    mask_preds,          # list[Tensor], each (1, H_l, W_l, S_l^2)
    mask_feat,          # Tensor [1, H_l, W_l, E]
    resized_image_shape, # (resized_h, resized_w)
    score_threshold=0.5,
    from_logits=True,       # We'll interpret it as "model returns logits => apply sigmoid"
    nms_method='gaussian',
    include_background=False,
    nms_sigma=0.5
):
    """
    Post-processes SOLO network outputs to generate instance masks and class predictions.

    This function takes multi-scale classification and mask outputs from the SOLO model,
    applies thresholding, selects candidate masks, and performs matrix NMS (Non-Maximum Suppression)
    to filter and refine the final instances.

    Args:
        cate_preds (List[Tensor]): List of category score tensors, each of shape (1, S_l, S_l, C),
            where S_l is the grid size at level l, and C is the number of classes.
        mask_preds (List[Tensor]): List of mask prediction tensors, each of shape (1, H_l, W_l, S_l),
            where H_l and W_l are the spatial dimensions of the feature maps.
        mask_feat (Tensor): Mask feature tensor of shape (1, H_l, W_l, E), where E is the feature dimension.
        resized_image_shape (Tuple[int, int]): Target shape (height, width) of the output masks.
        score_threshold (float, optional): Minimum confidence threshold to retain a (cell, class) prediction. Defaults to 0.5.
        from_logits (bool, optional): Whether the `cate_preds` are logits (if True, sigmoid will be applied). Defaults to True.
        nms_method (str, optional): NMS method to use ("gaussian" or "linear"). Defaults to "gaussian".
        include_background (bool, optional): Whether to include background class predictions. Defaults to False.
        nms_sigma (float, optional): Sigma value for Gaussian NMS method. Defaults to 0.5.

    Returns:
        List[dict]: A list of instances, each a dictionary containing:
            - "class_id" (int): Predicted class ID.
            - "score" (float): Confidence score after NMS.
            - "mask" (np.ndarray): Binary mask (shape: resized_h x resized_w) with values in {0, 1}.

    Notes:
        - Soft masks are generated before binarization after NMS.
        - Multiple predictions can initially originate from the same grid cell.
        - Matrix NMS is used for better handling of overlapping masks compared to standard NMS.
    """
    resized_h, resized_w = resized_image_shape

    # Gather all candidates
    all_scores = []
    all_classes = []
    all_masks_resized = []

    # batch size
    B = tf.shape(mask_feat)[0]
    # featuremap spatial dims
    H, W = tf.shape(mask_feat)[1], tf.shape(mask_feat)[2]
    # number of kernels
    E = tf.shape(mask_feat)[3]

    # Flatten mask_feat spatially: [B, H*W, E]
    mask_feat_flat = tf.reshape(mask_feat, [B, H * W, E])

    # Loop over FPN levels
    for cate_out, mask_kernel in zip(cate_preds, mask_preds):
        # Remove batch dim => (S_l, S_l, C) & (H_l, W_l, S_l^2)
        cate_out = cate_out[0]  # => shape (S_l, S_l, C)

        # Flatten mask_kernel_pred over the grid to get [B, S^2, E]
        mask_kernel_flat = tf.reshape(mask_kernel, [B, -1, E])
        # Dynamic conv: dot each kernel with every spatial feature
        #    mask_pred: [B, H*W, N]
        mask_out = tf.linalg.matmul(mask_feat_flat, mask_kernel_flat, transpose_b=True)
        mask_out = tf.reshape(mask_out, (B, H, W, tf.shape(mask_out)[2]))


        mask_out = mask_out[0]  # => shape (H_l, W_l, S_l^2)
        S_l = cate_out.shape[0]

        # Convert category logits => per-class probabilities (sigmoid)
        if from_logits:
            cate_prob = tf.sigmoid(cate_out).numpy()  # shape (S_l, S_l, C)
        else:
            cate_prob = cate_out.numpy()  # already in [0,1]

        # Convert mask logits => [0,1]
        if from_logits:
            mask_prob = tf.sigmoid(mask_out).numpy()  # shape (H_l, W_l, S_l^2)
        else:
            mask_prob = mask_out.numpy()

        if not include_background:
            # Get rid of the background. Assume background has 0th index
            cate_prob = cate_prob[..., 1:]

        # For each cell (gy, gx), find all classes above score_threshold
        for gy in range(S_l):
            for gx in range(S_l):
                class_probs = cate_prob[gy, gx, :]  # shape (C,)
                # Multi-label approach => find all classes with prob >= threshold
                above_thresh_indices = np.where(class_probs >= score_threshold)[0]
                if len(above_thresh_indices) == 0:
                    continue

                # For each class that passes the threshold
                for cls_id in above_thresh_indices:
                    sc = class_probs[cls_id]
                    chan_idx = gy * S_l + gx
                    # Retrieve mask channel => shape (H_l, W_l)
                    small_mask = mask_prob[..., chan_idx].astype(np.float32)

                    # Upsample to (resized_h, resized_w)
                    up_mask = cv2.resize(
                        small_mask,
                        (resized_w, resized_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                    # Keep it soft in [0,1]
                    all_masks_resized.append(up_mask)
                    all_scores.append(sc)
                    all_classes.append(cls_id)

    # If no candidates, return empty
    if len(all_scores) == 0:
        return []

    # Convert to arrays for Matrix NMS
    all_scores_arr = np.array(all_scores, dtype=np.float32)     # (N,)
    all_classes_arr = np.array(all_classes, dtype=np.int32)     # (N,)
    all_masks_arr = np.stack(all_masks_resized, axis=0)         # (N, resized_h, resized_w)

    all_scores_arr_tf = tf.convert_to_tensor(all_scores_arr)
    all_classes_arr_tf = tf.convert_to_tensor(all_classes_arr)
    all_masks_arr_tf = tf.convert_to_tensor(all_masks_arr)

    # Matrix NMS
    kept_indices = matrix_nms(
        masks=all_masks_arr_tf,
        scores=all_scores_arr_tf,
        labels=all_classes_arr_tf,
        post_nms_k=None,
        score_threshold=score_threshold,
        sigma=nms_sigma
    )

    final_masks = all_masks_arr[kept_indices.numpy()]
    final_scores = all_scores_arr[kept_indices.numpy()]
    final_classes = all_classes_arr[kept_indices.numpy()]
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
    Overlays instance masks that are sized for 'resized_shape'
    onto 'original_image' which is bigger (orig_h, orig_w).

    Args:
        original_image (np.ndarray): shape (orig_h, orig_w, 3)
        instances (list of dict): each with {"class_id", "score", "mask"}
            where 'mask' is shape (resized_h, resized_w) in {0,1}
        show_labels (bool): show the labels of the instances if the flag set to True. Default True
        class_names: optional

    Returns:
        vis_image: same shape as original_image, with masks overlaid
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
                label_str = f"{class_names.get(inst['class_id'])}: {score_str}"
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

    solo = keras.models.load_model(model_path)

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
            include_background=cfg.include_background,
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


