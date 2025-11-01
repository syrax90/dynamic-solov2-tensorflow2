"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script performs the main actions for the training process.
"""


import os
import logging
import re
import tensorflow as tf
import tensorflow.keras.layers as layers
from coco_dataset import create_coco_tf_dataset, get_classes
from coco_dataset_optimized import create_coco_tfrecord_dataset
from config import DynamicSOLOConfig
from model_functions import SOLOModel
from loss import compute_multiscale_loss


FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

tf.keras.backend.clear_session()

# Enable dynamic memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

def extract_epoch_number(file_name):
    """
    Extract the epoch number from a saved Keras weight filename.
    Expects filenames that match the pattern `"...epoch{N}.keras"`.

    Args:
      file_name (str): Filename to parse, e.g. `"solo_epoch00000042.keras"`.

    Returns:
      int: The extracted epoch number.
    """
    return int(re.search(r'epoch(\d+)\.keras', file_name).group(1))

def train_one_epoch(model, dataset, optimizer, num_classes):
    """
    Run one training epoch without gradient accumulation.

    Iterates over the dataset once, computes loss, applies gradients, and prints
    per-step metrics.

    Args:
      model (tf.keras.Model): SOLO model whose forward pass returns
        `(class_outputs, mask_outputs, mask_feat)`, where:
          * `class_outputs`: `[B, sum(S_i^2) num_classes]`
          * `mask_outputs`: `[B, sum(S_i^2), D]`
          * `mask_feat`: `[B, H, W, D]`
        (`S` varies per FPN scale; `D` is the mask feature dimension).
      dataset (tf.data.Dataset): Yields triples
        `(images, cate_target, mask_target)` per step where:
          * `images`: tf.float32, `[B, H, W, 3]`.
          * `cate_target` (aka `class_target`): tf.int32 class indices per grid cell, `[B, sum(S_i^2)]`.
          * `mask_target`: tf.uint8 GT masks aligned to grid cells, `[B, H, W, sum(S_i^2)]`.
      optimizer (tf.keras.optimizers.Optimizer): Optimizer to update weights.
      num_classes (int): Number of object classes (background excluded).

    Returns:
      None
    """
    for step, (images, cate_target, mask_target)  in enumerate(dataset):
        total_loss, cate_loss, mask_loss = train_one_step(model, images, cate_target, mask_target, optimizer, num_classes)
        print("Step ", step, ": ", "total=", total_loss.numpy(), ", cate=", cate_loss.numpy(), ", mask=", mask_loss.numpy())

@tf.function
def train_one_step(model, images, cate_target, mask_target, optimizer, num_classes):
    """
    Perform a single optimization step (no accumulation).

    Runs a forward pass, computes multiscale SOLO losses, backpropagates, and
    applies gradients.

    Args:
      model (tf.keras.Model): SOLO model producing:
        * `class_outputs`: `[B, sum(S_i^2), num_classes]`
        * `mask_outputs`: `[B, sum(S_i^2), D]`
        * `mask_feat`: `[B, H, W, D]`.
      images (tf.Tensor): float32 input images, `[B, H, W, 3]`.
      cate_target (tf.Tensor): Class indices per grid cell (a.k.a.
        `class_target`), `[B, sum(S_i^2)]`.
      mask_target (tf.Tensor): GT masks aligned to grid cells,
        `[B, H, W, sum(S_i^2)]`.
      optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.
      num_classes (int | tf.Tensor): Number of classes (background excluded).

    Returns:
      Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        * `total_loss`: scalar total loss.
        * `cate_loss`: scalar classification (focal) loss.
        * `mask_loss`: scalar mask (Dice) loss.
    """
    with tf.GradientTape() as tape:
        class_outputs, mask_outputs, mask_feat = model(images, training=True)
        total_loss, cate_loss, mask_loss = compute_multiscale_loss(
            class_outputs, mask_outputs, mask_feat,
            cate_target, mask_target,
            num_classes=tf.constant(num_classes))
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, cate_loss, mask_loss

@tf.function
def accumulate_one_step(model,
                        images,
                        cate_target,
                        mask_target,
                        num_classes,
                        accum_grads):
    """
    Accumulate gradients for one mini-batch (no optimizer step).

    Computes SOLO multiscale losses and adds gradients into preallocated
    buffers to enable gradient accumulation across multiple steps.

    Args:
      model (tf.keras.Model): SOLO model producing:
        * `class_outputs`: `[B, sum(S_i^2), num_classes]`
        * `mask_outputs`: `[B, sum(S_i^2), D]`
        * `mask_feat`: `[B, H, W, D]`.
      images (tf.Tensor): float32 `[B, H, W, 3]`.
      cate_target (tf.Tensor): Class indices `[B, sum(S_i^2)]`.
      mask_target (tf.Tensor): GT masks `[B, H, W, sum(S_i^2)]`.
      num_classes (int | tf.Tensor): Number of classes (background excluded).
      accum_grads (List[tf.Variable]): Zero-initialized gradient buffers, one per entry in `model.trainable_variables`; same shapes/dtypes.

    Returns:
      Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        * `total_l`: scalar total loss.
        * `cate_l`: scalar classification loss.
        * `mask_l`: scalar mask loss.
    """
    with tf.GradientTape() as tape:
        class_outputs, mask_outputs, mask_feat = model(images, training=True)
        total_l, cate_l, mask_l = compute_multiscale_loss(
            class_outputs, mask_outputs, mask_feat,
            cate_target, mask_target,
            num_classes=tf.constant(num_classes)
        )

    grads = tape.gradient(total_l, model.trainable_variables)

    # add to buffers
    for g_acc, g in zip(accum_grads, grads):
        if g is not None:
            g_acc.assign_add(g)

    return total_l, cate_l, mask_l

@tf.function
def train_one_epoch_accumulated_mode(model,
                    dataset,
                    optimizer,
                    num_classes,
                    accumulation_steps,
                    accum_grads,
                    accum_counter,
                    global_step):
    """
    Run one epoch with gradient accumulation using preallocated buffers.

    For each batch, compute SOLO multiscale losses and add gradients to
    `accum_grads`. Apply an optimizer step every `accumulation_steps` by
    dividing buffered grads by `accumulation_steps` and resetting buffers.
    Any leftover buffered grads at epoch end are cleared (no update).

    Args:
      model (tf.keras.Model): SOLO model producing:
        * `class_outputs`: `[B, sum(S_i^2), num_classes]`
        * `mask_outputs`: `[B, sum(S_i^2), D]`
        * `mask_feat`: `[B, H, W, D]`.
      dataset (tf.data.Dataset): Yields
        `(images, cate_target, mask_target)` with shapes:
        * `images`: `[B, H, W, 3]` (float32)
        * `cate_target`: `[B, sum(S_i^2)]`
        * `mask_target`: `[B, H, W, sum(S_i^2)]`.
      optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.
      num_classes (int | tf.Tensor): Number of classes (background excluded).
      accumulation_steps (int): Number of mini-batches to accumulate.
      accum_grads (List[tf.Variable]): Gradient buffers (zeros, same shapes as `model.trainable_variables`).
      accum_counter (tf.Variable): int32 counter tracking steps since last apply.
      global_step (tf.Variable): int32 counter of total steps in the epoch.

    Returns:
      None
    """

    # helper: apply optimiser + reset buffers
    def _apply_and_reset(denominator):
        scaled = [g / tf.cast(denominator, g.dtype) for g in accum_grads]
        optimizer.apply_gradients(zip(scaled, model.trainable_variables))
        for g in accum_grads:
            g.assign(tf.zeros_like(g))
        accum_counter.assign(0)

    def _reset_counter():
        for g in accum_grads:
            g.assign(tf.zeros_like(g))
        accum_counter.assign(0)

    for images, cate_target, mask_target in dataset:                     # AutoGraph → while_loop
        tot, cat, msk = accumulate_one_step(
            model, images, cate_target, mask_target, num_classes, accum_grads)

        accum_counter.assign_add(1)
        global_step.assign_add(1)

        tf.cond(accum_counter == accumulation_steps,
                lambda: _apply_and_reset(accumulation_steps),
                lambda: None)

        tf.print("step", global_step,
                 ": total =", tot,
                 "cate =", cat,
                 "mask =", msk)

    # flush leftovers
    tf.cond(accum_counter > 0,
            lambda: _reset_counter(),
            lambda: None)

def run_one_epoch_accumulated_mode(model,
                  dataset,
                  optimizer,
                  num_classes,
                  accumulation_steps=8):
    """
    Convenience wrapper to run one epoch with gradient accumulation.

    Allocates zero-initialized gradient buffers and integer counters, then
    calls :func:`train_one_epoch_accumulated_mode`.

    Args:
      model (tf.keras.Model): SOLO model producing:
        * `class_outputs`: `[B, sum(S_i^2), num_classes]`
        * `mask_outputs`: `[B, sum(S_i^2), D]`
        * `mask_feat`: `[B, H, W, D]`.
      dataset (tf.data.Dataset): Yields
        `(images, cate_target, mask_target)` with shapes:
        * `images`: `[B, H, W, 3]` (float32)
        * `cate_target`: `[B, sum(S_i^2)]`
        * `mask_target`: `[B, H, W, sum(S_i^2)]`.
      optimizer (tf.keras.optimizers.Optimizer): Optimizer instance.
      num_classes (int): Number of classes (background excluded).
      accumulation_steps (int, optional): Steps to accumulate before applying updates. Defaults to `8`.

    Returns:
      None
    """
    # gradient buffers (one per trainable weight)
    accum_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False)
        for v in model.trainable_variables
    ]

    # counters
    accum_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    global_step   = tf.Variable(0, dtype=tf.int32, trainable=False)

    # run the compiled graph
    train_one_epoch_accumulated_mode(model,
                    dataset,
                    optimizer,
                    num_classes,
                    accumulation_steps,
                    accum_grads,
                    accum_counter,
                    global_step)


if __name__ == '__main__':
    cfg = DynamicSOLOConfig()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    batch_size = cfg.batch_size
    img_height, img_width = cfg.img_height, cfg.img_width

    previous_epoch = 0
    # load previous model if load_previous_model = True
    load_previous_model = cfg.load_previous_model
    if load_previous_model:
        # Create a model instance because of NGC's TensorFlow version
        solo_model = SOLOModel(
            input_shape=(img_height, img_width, 3),  # Example shape
            num_classes=num_classes,
            num_stacked_convs=7,
            head_input_channels=256,
            mask_kernel_channels=256,
            grid_sizes=cfg.grid_sizes
        )
        solo_model.build((None, img_height, img_width, 3))
        solo_model.load_weights(cfg.model_path)
        previous_epoch = extract_epoch_number(cfg.model_path)
    else:
        solo_model = SOLOModel(
            input_shape=(img_height, img_width, 3),  # Example shape
            num_classes=num_classes,
            num_stacked_convs=7,
            head_input_channels=256,
            mask_kernel_channels=256,
            grid_sizes=cfg.grid_sizes
        )
        solo_model.build((None, img_height, img_width, 3))

    if previous_epoch > cfg.epochs:
        print(f'The model is trained {previous_epoch} epochs already while configuration assumes {cfg.epochs} epochs.')
        exit(0)

    # Form COCO dataset
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
            number_images=cfg.number_images,
            augment=cfg.augment
        )

    optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.lr, momentum=0.9)

    # Training loop
    print("Starting training...")
    for epoch in range(previous_epoch + 1, cfg.epochs):
        print(f"Starting epoch {epoch}:")
        if cfg.use_gradient_accumulation_steps:
            run_one_epoch_accumulated_mode(solo_model, ds, optimizer, num_classes, accumulation_steps=cfg.accumulation_steps)
        else:
            train_one_epoch(solo_model, ds, optimizer, num_classes)
        # ==================== save ====================
        if epoch != 0 and epoch % cfg.save_iter == 0:
            save_path = f'./weights/{cfg.model_weights_prefix}_epoch%.8d.keras' % epoch
            solo_model.save(save_path)
            path_dir = os.listdir('./weights')
            epoch_numbers = []
            names = []
            for name in path_dir:
                if name.endswith('.keras') and name.startswith(cfg.model_weights_prefix):
                    epoch_number = extract_epoch_number(name)
                    epoch_numbers.append(epoch_number)
                    names.append(name)
            if len(epoch_numbers) > 10:
                i = epoch_numbers.index(min(epoch_numbers))
                os.remove('./weights/' + names[i])
            logger.info('Save model to {}'.format(save_path))
    print("Done!")
