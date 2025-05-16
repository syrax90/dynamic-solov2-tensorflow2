"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains configurations.
"""


class DynamicSOLOConfig(object):
    def __init__(self):
        self.coco_root_path = '/path/to/your/coco/dataset'
        self.train_annotation_path = f'{self.coco_root_path}/annotations/instances_train2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.images_path = f'{self.coco_root_path}/train2017/'
        self.include_background=True    # Exclude background 0th class for custom COCO dataset
        self.number_images=None  # Restriction for dataset. Set None to get rid of the restriction

        # Image parameters
        self.img_height = 320
        self.img_width = 320

        # If load_previous_model = True: load the previous model weights (example: './weights/coco_epoch00001000.keras')
        self.load_previous_model = False
        self.lr = 0.0001
        self.batch_size = 8
        # If load_previous_model = True, you need to specify self.model_path to indicate which model to read the weights from to continue training.
        self.model_path = './weights/coco_epoch00000001.keras'

        # Save the model weights every save_iter epochs:
        self.save_iter = 1
        # Number of epochs
        self.epochs = 30000
        # Model weights file prefix
        self.model_weights_prefix = 'coco'

        self.grid_sizes = [40, 36, 24, 16]
        self.image_scales = [0.25]

        # Testing configuration
        self.test_model_path = './weights/coco_epoch00000001.keras'
        self.score_threshold = 0.5