"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains functions for the loss calculation.
"""


import tensorflow as tf


def focal_loss(
        logits,  # [B, H, W, num_classes]
        targets,  # [B, H, W, num_classes]
        alpha=0.25,
        gamma=2.0
):
    """
    Computes focal loss for multi-class classification. It expects 'targets' to be one-hot.

    Args:
        logits:   [B, H, W, C]
        targets:  [B, H, W, C] one-hot
        alpha:    weighting factor for positive samples
        gamma:    exponent factor to down-weight easy examples

    Returns:
        A scalar Tensor of shape [] (the mean loss).
    """
    # Typically we do logits => prob using sigmoid or softmax
    # If it's purely multi-class with exactly 1 class active at each location,
    # you might prefer softmax. Let's assume "sigmoid" multi-label style for simplicity:
    probs = tf.nn.sigmoid(logits)  # shape [B, H, W, C]

    # The focal loss formula (for binary or multi-label) is:
    # FL = - alpha * targets*(1 - probs)^gamma * log(probs)
    #      - (1-alpha) * (1-targets)* probs^gamma * log(1 - probs)
    pt = tf.where(tf.equal(targets, 1.), probs, 1. - probs)
    focal_factor = alpha * tf.cast(tf.equal(targets, 1.), tf.float32) + \
                   (1. - alpha) * tf.cast(tf.equal(targets, 0.), tf.float32)
    focal_weight = focal_factor * tf.pow((1. - pt), gamma)

    # Cross-entropy part
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

    # Focal loss
    loss = focal_weight * ce_loss

    return tf.reduce_mean(loss)

def dice_loss(pred_mask, gt_mask, eps=1e-5):
    """
    Computes a per-instance Dice Loss for a predicted mask vs. ground truth.

    Args:
        pred_mask:  [B, H, W, S] with probabilities in [0,1]
        gt_mask:    [B, H, W, S] the same shape, with 0/1 ground-truth
        eps:        small constant to avoid division by zero

    Returns:
        Scalar dice loss (1 - dice_coefficient).
    """

    intersection = tf.reduce_sum(pred_mask * gt_mask, axis=[1, 2])
    union = tf.reduce_sum(tf.square(pred_mask), axis=[1, 2]) + tf.reduce_sum(tf.square(gt_mask), axis=[1, 2]) + eps
    dice_coef = (2.0 * intersection + eps) / union
    dice_loss_value = 1.0 - dice_coef

    # We only want the loss for positive instances (where a ground-truth mask exists).
    # Determine positive masks: a mask is positive if y_true has any foreground pixel.
    # (Alternatively, one could use the classification labels to identify positives.)
    pos_mask = tf.reduce_max(gt_mask, axis=[1, 2])  # shape [batch, S^2], 1.0 for any mask that has at least one pixel

    # `pos_mask` will be 1.0 for positive instances, 0.0 for background instances.
    # Multiply dice loss by pos_mask to ignore losses from negative (all-zero) masks.
    masked_dice_loss = dice_loss_value * pos_mask

    # Sum over all masks and batch
    loss_sum = tf.reduce_sum(masked_dice_loss)

    # Number of positive instances (masks) across the batch
    num_pos = tf.reduce_sum(pos_mask)

    # Average the dice loss only over positive instances
    loss = loss_sum / tf.maximum(num_pos, 1.0)
    return loss


def solo_loss(
        cls_pred,  # [B, S, S, num_classes]
        mask_pred,  # [B, H, W, S_i]
        cls_target,  # [B, S, S, num_classes]
        gt_masks_for_cells,  # [B, H, W, S_i]
        cls_loss_weight=1,
        mask_loss_weight=1,
):
    """
    Computes the SOLO loss for a single scale, combining classification and mask prediction losses.

    Args:
        cls_pred (Tensor): Predicted class scores with shape [B, S, S, num_classes],
            where B is the batch size, S is the grid size, and num_classes is the number of classes.
        mask_pred (Tensor): Predicted masks with shape [B, H, W, S_i],
            where H and W are spatial dimensions of P2 FPN level, and S_i corresponds to the number of masks per cell.
        cls_target (Tensor): Ground truth class labels with shape [B, S, S, num_classes].
        gt_masks_for_cells (Tensor): Ground truth masks aligned to cells with shape [B, H, W, S_i].
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
    mask_loss_value = dice_loss(mask_probs, gt_masks_valid)

    # Decrease mask_loss_value to train it faster
    total_loss = cls_loss_weight * cls_loss_value + mask_loss_weight * mask_loss_value
    return total_loss, cls_loss_value, mask_loss_value


def compute_multiscale_loss(outputs, targets, num_scales, num_classes):
    """
    Computes SOLO loss for multiple scales (FPN levels)

    Args:
        outputs: list of outputs:
            outputs[0]: list of categories tensors for each FPN level
                outputs[0][0]: [B, S0, S0, num_classes] For P2 FPN level
                ...
                outputs[0][3]: [B, S3, S3, num_classes] For P5 FPN level
            outputs[1]: list of kernel tensors for each FPN level
                outputs[0][0]: [B, S0, S0, D] For P2 FPN level
                ...
                outputs[0][3]: [B, S3, S3, D] For P5 FPN level
            outputs[2]: Mask feature tensor [B, H, W, D]    H and W equal to H and W for P2 FPN level

        targets: dict with keys like:
           "cate_target_0" -> shape [B, S0, S0] (integer labels, -1=ignore) For P2 FPN level
           "mask_target_0" -> shape [B, H, W, S0]  H and W equal to H and W for P2 FPN level
           ...
           "cate_target_3" -> shape [B, S3, S3] (integer labels, -1=ignore) For P5 FPN level
           "mask_target_3" -> shape [B, H, W, S3]  H and W equal to H and W for P2 FPN level

        num_scales: number of FPN levels
        num_classes: number of categories

    Returns:
       total_loss, total_cate_loss, total_mask_loss
    """
    total_loss = tf.constant(0.0)
    total_cate_loss = tf.constant(0.0)
    total_mask_loss = tf.constant(0.0)

    mask_feat_pred = outputs[2]  # [B, H, W, D]
    # batch size
    B = tf.shape(mask_feat_pred)[0]
    # featuremap spatial dims
    H, W = tf.shape(mask_feat_pred)[1], tf.shape(mask_feat_pred)[2]
    # number of kernel weights
    D = tf.shape(mask_feat_pred)[3]

    # Flatten mask_feat spatially: [B, H*W, D]
    mask_feat_flat = tf.reshape(mask_feat_pred, [B, H * W, D])

    for i in range(num_scales):
        # --------------------------------------------------------------
        # For Category Loss
        # --------------------------------------------------------------
        cate_pred = outputs[0][i]  # [B, S_i, S_i, num_classes]
        cate_true = targets[f"cate_target_{i}"]  # [B, S_i, S_i]
        cate_true = tf.one_hot(cate_true, depth=num_classes, axis=-1)

        # --------------------------------------------------------------
        # For Mask Loss
        # --------------------------------------------------------------
        mask_kernel_pred = outputs[1][i]  # [B, S_i, S_i, D]

        # Flatten mask_kernel_pred over the grid to get [B, S_i, D]
        mask_kernel_flat = tf.reshape(mask_kernel_pred, [B, -1, D])

        # Dynamic conv: dot each kernel with every spatial feature
        #    mask_pred: [B, H*W, N]
        mask_pred = tf.linalg.matmul(mask_feat_flat, mask_kernel_flat, transpose_b=True)
        mask_pred = tf.reshape(mask_pred, (B, H, W, tf.shape(mask_pred)[2]))    # [B, H, W, S_i]

        mask_true = targets[f"mask_target_{i}"]  # [B, H_i, W_i, S_i]

        # --------------------------------------------------------------
        # Calculating SOLO Loss
        # --------------------------------------------------------------
        total_scale_loss, scale_cate_loss, scale_mask_loss = solo_loss(
            cate_pred, mask_pred, cate_true, mask_true
        )
        total_loss += total_scale_loss
        total_cate_loss += scale_cate_loss
        total_mask_loss += scale_mask_loss

    # Average across the number of scales
    total_cate_loss /= num_scales
    total_mask_loss /= num_scales
    total_loss /= num_scales

    return total_loss, total_cate_loss, total_mask_loss