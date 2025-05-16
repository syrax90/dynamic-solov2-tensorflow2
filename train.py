"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script performs the main actions for the training process.
"""


import os
import keras
import logging
import re
import tensorflow as tf
import tensorflow.keras.layers as layers
from coco_dataset import create_coco_tf_dataset, get_classes
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
    return int(re.search(r'epoch(\d+)\.keras', file_name).group(1))

def train_one_epoch(model, dataset, optimizer, num_classes):
    """
    model: a tf.keras.Model (e.g., MultiScaleSOLO).
    dataset: a tf.data.Dataset yielding (images, multi_scale_targets).
    optimizer: e.g., tf.keras.optimizers.Adam or SGD.

    We'll compute the multi-scale loss and update model weights each step.
    """
    for step, (images, targets) in enumerate(dataset):
        total_loss, cate_loss, mask_loss = train_one_step(model, images, targets, optimizer, num_classes)
        print("Step ", step, ": ", "total=", total_loss.numpy(), ", cate=", cate_loss.numpy(), ", mask=", mask_loss.numpy())

@tf.function
def train_one_step(model, images, targets, optimizer, num_classes):
    """
    model: a tf.keras.Model (e.g., MultiScaleSOLO).
    dataset: a tf.data.Dataset yielding (images, multi_scale_targets).
    optimizer: e.g., tf.keras.optimizers.Adam or SGD.

    We'll compute the multi-scale loss and update model weights each step.
    """
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        total_loss, cate_loss, mask_loss = compute_multiscale_loss(
            outputs,
            targets,
            num_scales=int(len(targets) / 2),
            num_classes=tf.constant(num_classes))
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, cate_loss, mask_loss


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
        solo_model = keras.models.load_model(cfg.model_path)
        previous_epoch = extract_epoch_number(cfg.model_path)
    else:
        solo_model = SOLOModel(
            input_shape=(img_height, img_width, 3),  # Example shape
            num_classes=num_classes,
            num_stacked_convs=4,
            head_input_channels=256,
            mask_kernel_channels=256,
            grid_sizes=cfg.grid_sizes
        )

    if previous_epoch > cfg.epochs:
        print(f'The model is trained {previous_epoch} epochs already while configuration assumes {cfg.epochs} epochs.')
        exit(0)

    # Form COCO dataset
    ds = create_coco_tf_dataset(
        coco_annotation_file=cfg.train_annotation_path,
        coco_img_dir=cfg.images_path,
        num_classes=num_classes,
        target_size=(img_height, img_width),
        batch_size=cfg.batch_size,
        grid_sizes=cfg.grid_sizes,
        scale=cfg.image_scales[0],
        number_images=cfg.number_images
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(previous_epoch + 1, cfg.epochs):
        print(f"Starting epoch {epoch}:")
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
