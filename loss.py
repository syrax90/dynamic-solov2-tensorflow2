"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains functions for the loss calculation.
"""


import tensorflow as tf


def focal_loss(
        logits,   # [B, H, W, C]
        targets,  # [B, H, W, C] one-hot
        alpha=None,   # per-class weight for positives and negatives
        gamma=2.0,
        eps=1e-7
):
    """
    Softmax focal loss with explicit penalty for false positives (non-target classes).

    Args:
        logits:  [B, H, W, C] unnormalized scores.
        targets: [B, H, W, C] one-hot (exactly one 1 per pixel).
        alpha:   per-class weighting for positives and negatives. If None, uses [0.001 for background, 0.75 for others].
        gamma:   focusing parameter.
        eps:     numerical stability for logs.

    Returns:
        Scalar Tensor (mean loss).
    """
    num_classes = logits.shape[-1]

    # Default alpha: emphasize foreground vs background
    if alpha is None:
        alpha = [0.001] + [0.75] * (num_classes - 1)
    alpha = tf.constant(alpha, dtype=logits.dtype)  # [C]
    alpha = tf.cast(alpha, logits.dtype)        # positives

    # Softmax probabilities
    probs = tf.nn.softmax(logits, axis=-1)                 # [B,H,W,C]
    probs = tf.clip_by_value(probs, eps, 1.0 - eps)
    log_probs   = tf.math.log(probs)                       # [B,H,W,C]
    log1m_probs = tf.math.log1p(-probs)                    # [B,H,W,C]

    # Positive (true-class) focal term: -alpha * (1 - p)^gamma * log(p)
    pos_weight = tf.pow(1.0 - probs, gamma)
    pos_term = -targets * alpha * pos_weight * log_probs

    # Negative (non-true-class) focal term (penalizes false positives):
    # For each non-target class k: -alpha * p_k^gamma * log(1 - p_k)
    neg_weight = tf.pow(probs, gamma)
    neg_term = -(1.0 - targets) * alpha * neg_weight * log1m_probs

    # Sum over classes, then mean over batch/spatial dims
    loss = tf.reduce_sum(pos_term + neg_term, axis=-1)     # [B,H,W]
    return tf.reduce_mean(loss)


def dice_loss(pred_mask, gt_mask, cls_target, eps=1e-5):
    """
    Computes per-instance Dice Loss for predicted vs. ground-truth masks,
    considering only positive cells (as in SOLO).

    Args:
        pred_mask:  [B, H, W, S^2] with probabilities in [0,1]
        gt_mask:    [B, H, W, S^2] the same shape, with 0/1 ground-truth
        cls_target: [B, sum(S_i^2), num_classes] ground truth class labels (one-hot)
        eps:        small constant to avoid division by zero

    Returns:
        Scalar dice loss for positive cells.
    """
    # Convert cls_target to positive mask [B, S^2]
    # If it's one-hot, sum across classes → positive cell indicator
    pos_mask = tf.reduce_sum(cls_target, axis=-1)  # [B, S, S]
    pos_mask = tf.reshape(pos_mask, [tf.shape(pos_mask)[0], -1])  # [B, S^2]

    # Flatten masks for batch processing
    pred_mask = tf.reshape(pred_mask, [tf.shape(pred_mask)[0], -1, tf.shape(pred_mask)[-1]])  # [B, HW, S^2]
    gt_mask   = tf.reshape(gt_mask,   [tf.shape(gt_mask)[0], -1, tf.shape(gt_mask)[-1]])    # [B, HW, S^2]

    # pos_mask: [B, S^2] {0,1} → bool
    pos_bool = tf.cast(pos_mask > 0, tf.bool)  # [B, S^2]
    HW = tf.shape(pred_mask)[1]

    mask3 = tf.tile(pos_bool[:, tf.newaxis, :], [1, HW, 1])  # [B, HW, S]

    pred_mask = tf.ragged.boolean_mask(pred_mask, mask3)
    gt_mask = tf.ragged.boolean_mask(gt_mask, mask3)

    # Compute intersection and union
    intersection = tf.reduce_sum(pred_mask * gt_mask, axis=1)   # [B, S^2]
    union = tf.reduce_sum(tf.square(pred_mask), axis=1) + tf.reduce_sum(tf.square(gt_mask), axis=1) + eps

    dice_coef = (2.0 * intersection + eps) / union  # [B, S^2]

    # Apply only positive cells
    dice_loss_value = 1.0 - dice_coef
    dice_loss_value = tf.reduce_sum(dice_loss_value) / (tf.reduce_sum(pos_mask) + eps)

    return dice_loss_value


def solo_loss(
        cls_pred,  # [B, sum(S_i^2), num_classes]
        mask_pred,  # [B, H, W, sum(S_i^2)]
        cls_target,  # [B, sum(S_i^2), num_classes]
        gt_masks_for_cells,  # [B, H, W, sum(S_i^2)]
        cls_loss_weight=1,
        mask_loss_weight=1,
):
    """
    Computes the SOLO loss for a single scale, combining classification and mask prediction losses.

    Args:
        cls_pred (Tensor): Predicted class scores with shape [B, sum(S_i^2), num_classes],
            where `B` is the batch size, S_i is the grid size of corresponding FPN level, and num_classes is the number of classes.
        mask_pred (Tensor): Predicted masks with shape [B, H, W, sum(S_i^2)],
            where H and W are spatial dimensions of P2 FPN level, and S_i corresponds to the number of masks per cell.
        cls_target (Tensor): Ground truth class labels with shape [B, sum(S_i^2), num_classes].
        gt_masks_for_cells (Tensor): Ground truth masks aligned to cells with shape [B, H, W, sum(S_i^2)].
        cls_loss_weight (int): regularization weight of classification loss.
        mask_loss_weight (int): regularization weight of mask prediction loss.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - total_loss (Tensor): The combined loss from classification and mask prediction.
            - cls_loss_value (Tensor): The classification loss component.
            - mask_loss_value (Tensor): The mask prediction loss component.
    """
    # Classification loss
    cls_loss_value = focal_loss(cls_pred, cls_target)

    mask_probs = tf.nn.sigmoid(mask_pred)

    # Compute dice loss for all masks in a vectorized way
    gt_masks_valid = tf.cast(gt_masks_for_cells, tf.float32)

    cls_target_no_bg = cls_target[..., 1:]  # Exclude background class
    mask_loss_value = dice_loss(mask_probs, gt_masks_valid, cls_target_no_bg)

    # Decrease mask_loss_value to train it faster
    total_loss = cls_loss_weight * cls_loss_value + mask_loss_weight * mask_loss_value
    return total_loss, cls_loss_value, mask_loss_value


def compute_multiscale_loss(class_outputs, mask_outputs, mask_feat, class_target, mask_target, num_classes):
    """
    Computes the total SOLO loss across multiple feature scales, combining classification and mask prediction losses.

    This function integrates the classification (focal) loss and the mask (Dice) loss
    for a given feature pyramid level. It prepares one-hot targets, reshapes feature maps,
    computes mask predictions, and calls `solo_loss()` to compute the combined loss.

    Args:
        class_outputs (Tensor):
            Predicted class scores for each grid cell with shape `[B, sum(S_i^2), num_classes]`,
            where `B` is the batch size, `S_i` is the grid size of corresponding FPN level, and `num_classes` is
            the number of object categories (excluding background).
        mask_outputs (Tensor):
            Mask coefficient predictions for each grid cell, typically with shape `[B, sum(S_i^2), D]`,
            where `S_i` varies per scale and `D` is the feature dimension.
        mask_feat (Tensor):
            Feature map used for mask generation, with shape `[B, H, W, D]`, where
            `H` and `W` are spatial dimensions and `D` is the feature dimension.
        class_target (Tensor):
            Ground-truth class indices for each grid cell, with shape `[B, sum(S_i^2)]`.
        mask_target (Tensor):
            Ground-truth masks aligned to grid cells, with shape `[B, H, W, sum(S_i^2)]`.
        num_classes (int):
            Number of object classes (excluding background).

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            A tuple containing:

            - **total_loss** (`Tensor`): The combined total loss (classification + mask losses).
            - **total_cate_loss** (`Tensor`): The classification (focal) loss component.
            - **total_mask_loss** (`Tensor`): The mask (Dice) loss component.

    Notes:
        - This function assumes the SOLO (Segmenting Objects by Locations) architecture.
        - The `mask_outputs` are used to linearly combine with `mask_feat` to produce
          instance-specific mask predictions before applying the Dice loss.
        - The classification loss is weighted more heavily (`cls_loss_weight=10`) to
          stabilize training in early iterations.
    """

    class_true_one_hot = tf.one_hot(class_target + 1, depth=num_classes + 1, axis=-1)

    mask_feat_flat = tf.reshape(mask_feat, [-1, tf.shape(mask_feat)[1] * tf.shape(mask_feat)[2],
                                            tf.shape(mask_feat)[3]])  # [B, H*W, D]
    # batch size
    B = tf.shape(mask_feat)[0]
    # featuremap spatial dims
    H, W = tf.shape(mask_feat)[1], tf.shape(mask_feat)[2]

    #    mask_pred: [B, H*W, sum(S_i*S_i)]
    mask_pred = tf.linalg.matmul(mask_feat_flat, mask_outputs, transpose_b=True)
    mask_pred = tf.reshape(mask_pred, (B, H, W, tf.shape(mask_pred)[2]))  # [B, H, W, sum(S_i*S_i)]

    # --------------------------------------------------------------
    # Calculating SOLO Loss
    # --------------------------------------------------------------
    total_loss, total_cate_loss, total_mask_loss = solo_loss(
        class_outputs, mask_pred, class_true_one_hot, mask_target, mask_loss_weight=1, cls_loss_weight=10
    )

    return total_loss, total_cate_loss, total_mask_loss