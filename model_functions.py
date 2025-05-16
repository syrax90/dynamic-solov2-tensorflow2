"""
Author: Pavel Timonin
Created: 2025-04-24
Description: This script contains classes and functions for SOLO model building.
"""


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, Add
from tensorflow.keras.models import Model

def build_fpn_resnet50(input_shape=(224, 224, 3), ouput_kernels_number=256):
    """
        Builds a Feature Pyramid Network (FPN) on top of a ResNet50 backbone.

        This function extracts feature maps from different stages of ResNet50 and constructs
        a feature pyramid with lateral and smoothing connections for multi-scale feature representation.

        Args:
            input_shape (tuple): The shape of the input images.
            ouput_kernels_number (int): Number of output channels for each pyramid level.

        Returns:
            tf.keras.Model: A Keras model with inputs as the image tensor and outputs as a list
            of feature maps [p2, p3, p4, p5] from different FPN levels.
    """
    # Load the ResNet50 model without the top classification layer
    backbone = ResNet50(include_top=False, input_shape=input_shape)
    backbone.trainable = True

    # Define the layers to extract features from
    c2_output = backbone.get_layer("conv2_block3_out").output
    c3_output = backbone.get_layer("conv3_block4_out").output
    c4_output = backbone.get_layer("conv4_block6_out").output
    c5_output = backbone.get_layer("conv5_block3_out").output

    # Lateral connections
    p5 = Conv2D(256, (1, 1), padding='same')(c5_output)
    p4 = Add()([
        UpSampling2D()(p5),
        Conv2D(256, (1, 1), padding='same')(c4_output)
    ])
    p3 = Add()([
        UpSampling2D()(p4),
        Conv2D(256, (1, 1), padding='same')(c3_output)
    ])
    p2 = Add()([
        UpSampling2D()(p3),
        Conv2D(256, (1, 1), padding='same')(c2_output)
    ])

    # Smoothing layers
    p5 = Conv2D(ouput_kernels_number, (3, 3), padding='same')(p5)
    p4 = Conv2D(ouput_kernels_number, (3, 3), padding='same')(p4)
    p3 = Conv2D(ouput_kernels_number, (3, 3), padding='same')(p3)
    p2 = Conv2D(ouput_kernels_number, (3, 3), padding='same')(p2)

    # Create the model
    return Model(inputs=backbone.input, outputs=[p2, p3, p4, p5])


def append_coord_channels(features):
    """
    Append (x, y) coordinate channels to `features`.

    Args:
        features (Tensor): features of shape [B, H, W, C]

    Returns:
        The same tensor with added pixel coordinates, that are normalized to [-1, 1] shape [B, H, W, C+2]
    """
    shape = tf.shape(features)
    batch_size, height, width = shape[0], shape[1], shape[2]

    # Create a grid of x coordinates in [-1, 1], shape [width]
    x_coords = tf.linspace(-1.0, 1.0, width)
    # Create a grid of y coordinates in [-1, 1], shape [height]
    y_coords = tf.linspace(-1.0, 1.0, height)

    # Meshgrid => xgrid,ygrid of shape [height, width]
    xgrid, ygrid = tf.meshgrid(x_coords, y_coords)

    # Expand dims => [height, width, 1]
    xgrid = tf.expand_dims(xgrid, axis=-1)
    ygrid = tf.expand_dims(ygrid, axis=-1)

    # Tile to match batch => [B, H, W, 1]
    xgrid = tf.tile(tf.expand_dims(xgrid, 0), [batch_size, 1, 1, 1])
    ygrid = tf.tile(tf.expand_dims(ygrid, 0), [batch_size, 1, 1, 1])

    # Concat => [B, H, W, 2]
    coords = tf.concat([xgrid, ygrid], axis=-1)

    # Concat with original features => [B, H, W, C+2]
    features_with_coords = tf.concat([features, coords], axis=-1)
    return features_with_coords


class ConvGNReLU(tf.keras.layers.Layer):
    """
    A block performing 3x3 conv -> group norm -> ReLU.
    """
    def __init__(self, out_channels, kernel_size=3, groups=32):
        """
       Initializes the ConvGNReLU block.

       Args:
           out_channels (int): Number of output channels for the convolution layer.
           kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
           groups (int, optional): Number of groups for Group Normalization. Defaults to 32.
        """
        super(ConvGNReLU, self).__init__()
        self.conv = Conv2D(
            out_channels,
            kernel_size,
            padding='same',
            use_bias=False
        )
        self.gn = tf.keras.layers.GroupNormalization(groups=groups, axis=-1)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=None, **kwargs):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class MaskFeatureFusion(tf.keras.layers.Layer):
    """
    Fuses multi-level FPN features to produce a unified mask feature map.
    - out_channels: number of channels for the mask feature map (denoted as E).
    """
    def __init__(self, out_channels=128, groups=32, use_coord_conv=True):
        """
        Initializes the MaskFeatureFusion layer.

        Args:
            out_channels (int, optional): Number of output channels for the mask feature map. Defaults to 128.
            groups (int, optional): Number of groups for Group Normalization. Defaults to 32.
            use_coord_conv (bool, optional): Flag indicating whether to use coordinate convolution. Defaults to True.
        """
        super(MaskFeatureFusion, self).__init__()

        self.use_coord_conv = use_coord_conv

        def make_conv_block():
            return tf.keras.Sequential([
                ConvGNReLU(out_channels, kernel_size=3, groups=groups),
                ConvGNReLU(out_channels, kernel_size=3, groups=groups),
                ConvGNReLU(out_channels, kernel_size=3, groups=groups),
                ConvGNReLU(out_channels, kernel_size=3, groups=groups),
            ])

        self.conv_P5 = make_conv_block()
        self.conv_P4 = make_conv_block()
        self.conv_P3 = make_conv_block()
        self.conv_P2 = make_conv_block()
        # Convs after merging each level
        self.merge_conv_P4 = ConvGNReLU(out_channels, kernel_size=3, groups=groups)
        self.merge_conv_P3 = ConvGNReLU(out_channels, kernel_size=3, groups=groups)
        self.merge_conv_P2 = ConvGNReLU(out_channels, kernel_size=3, groups=groups)
        # Final 1x1 conv to output the mask feature map
        self.conv_fuse = ConvGNReLU(out_channels, kernel_size=1, groups=groups)

    def call(self, fpn_features, training=None, **kwargs):
        P2, P3, P4, P5 = fpn_features  # expect a list of FPN feature maps
        if self.use_coord_conv:
            P5 = append_coord_channels(P5)
        # Process P5 and upsample to P4's size
        P5_upsampled = tf.image.resize(self.conv_P5(P5, training=training), size=tf.shape(P4)[1:3], method='bilinear')
        # Add to P4, then refine and upsample to P3's size
        P4_merged = self.merge_conv_P4(self.conv_P4(P4, training=training) + P5_upsampled, training=training)
        P4_upsampled = tf.image.resize(P4_merged, size=tf.shape(P3)[1:3], method='bilinear')
        # Add to P3, refine and upsample to P2's size
        P3_merged = self.merge_conv_P3(self.conv_P3(P3, training=training) + P4_upsampled, training=training)
        P3_upsampled = tf.image.resize(P3_merged, size=tf.shape(P2)[1:3], method='bilinear')
        # Add to P2 and refine
        P2_merged = self.merge_conv_P2(self.conv_P2(P2, training=training) + P3_upsampled, training=training)
        # Final 1x1 conv to produce the unified mask feature map (at P2 resolution, 1/4 of input)
        mask_feature_map = self.conv_fuse(P2_merged, training=training)
        return mask_feature_map


class DynamicSOLOHead(tf.keras.layers.Layer):
    """
    Dynamic SOLO head.

    This layer generates dynamic convolutional kernels and classification scores
    for each grid cell in a feature map, enabling per-instance mask prediction.
    """
    def __init__(self,
                 num_classes,
                 grid_size,
                 num_stacked_convs=4,
                 mask_kernel_channels=256,
                 use_coordconv=False,
                 name='VanillaSOLOHead',
                 **kwargs):
        """
        Initializes the DynamicSOLOHead layer.

        Args:
            num_classes (int): Number of object classes for classification.
            grid_size (int): Grid size to divide the image into cells for instance prediction.
            num_stacked_convs (int, optional): Number of stacked convolutions in each branch. Defaults to 4.
            mask_kernel_channels (int, optional): Number of output channels for the mask kernel generation. Defaults to 256.
            use_coordconv (bool, optional): Whether to include coordinate channels in the convolution. Defaults to False.
            name (str, optional): Name for the layer. Defaults to 'VanillaSOLOHead'.
            **kwargs: Additional keyword arguments for the base Layer class.
        """
        super(DynamicSOLOHead, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.use_coordconv = use_coordconv

        # Resize features to grid size
        self.resize_layer = tf.keras.layers.Resizing(grid_size, grid_size, interpolation='bilinear',
                                                     name=f'{name}_resize')

        # ----------------------------------------------------------------------
        # Classification branch
        # ----------------------------------------------------------------------
        # We build each layer as a small sub-network:
        #    Conv2D -> GroupNorm -> ReLU
        self.cls_convs = []
        for i in range(num_stacked_convs):
            self.cls_convs.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        256,
                        kernel_size=3,
                        padding='same',
                        use_bias=False,  # Because BN will handle the bias term
                        name=f'{name}_cls_conv_{i}'
                    ),
                    tf.keras.layers.GroupNormalization(name=f'{name}_cls_bn_{i}'),
                    tf.keras.layers.ReLU(name=f'{name}_cls_relu_{i}')
                ], name=f'{name}_cls_block_{i}')
            )

        self.cls_logits = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding='same', name=f'{name}_cls_logits')

        # ----------------------------------------------------------------------
        # Mask branch
        # ----------------------------------------------------------------------
        self.mask_convs = []
        for i in range(num_stacked_convs):
            self.mask_convs.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        256,
                        kernel_size=3,
                        padding='same',
                        use_bias=False,
                        name=f'{name}_mask_conv_{i}'
                    ),
                    tf.keras.layers.GroupNormalization(name=f'{name}_mask_bn_{i}'),
                    tf.keras.layers.ReLU(name=f'{name}_mask_relu_{i}')
                ], name=f'{name}_mask_block_{i}')
            )

        self.mask_logits = tf.keras.layers.Conv2D(
            filters=mask_kernel_channels,
            kernel_size=3,
            padding='same',
            name=f'{name}_mask_logits'
        )

    def call(self, fpn_feature, training=False):
        # Resize to grid size
        x_resized = self.resize_layer(fpn_feature)

        # Classification branch
        x_cls = x_resized
        for seq_block in self.cls_convs:
            x_cls = seq_block(x_cls, training=training)

        cls_output = self.cls_logits(x_cls, training=training)  # [B, S, S, num_classes]

        # Mask branch
        x = x_resized
        if self.use_coordconv:
            x = append_coord_channels(x)
        x_mask = x

        for seq_block in self.mask_convs:
            x_mask = seq_block(x_mask, training=training)

        mask_output = self.mask_logits(x_mask, training=training)  # [B, S, S, E]

        return cls_output, mask_output


class SOLOModel(tf.keras.Model):
    """
    SOLO (Segmenting Objects by Locations) model.

    This model integrates a backbone, FPN, and dynamic heads to perform instance segmentation
    by predicting object masks directly at different grid levels.
    """
    def __init__(
        self,
        input_shape=(None, None, 3),
        num_classes=80,
        grid_sizes=[40, 36, 24, 16],
        num_stacked_convs=1,
        head_input_channels=256,
        mask_kernel_channels=256,
        **kwargs
    ):
        """
        Initializes the SOLOModel.

        Args:
            input_shape (tuple, optional): Shape of the input images. Defaults to (None, None, 3).
            num_classes (int, optional): Number of object classes. Defaults to 80.
            grid_sizes (list, optional): Grid sizes for different feature levels. Defaults to [40, 36, 24, 16].
            num_stacked_convs (int, optional): Number of stacked conv layers in the SOLO head. Defaults to 1.
            head_input_channels (int, optional): Input channels for the SOLO head. Defaults to 256.
            mask_kernel_channels (int, optional): Output channels for mask kernel conv layers. Defaults to 256.
            **kwargs: Additional keyword arguments for the base Model class.
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_scales = len(grid_sizes)

        # Build FPN-ResNet backbone
        self.backbone = build_fpn_resnet50(input_shape, ouput_kernels_number=head_input_channels)

        # Build the head layer for each scale
        self.solo_heads = []
        for i in range(self.num_scales):
            head = DynamicSOLOHead(
                num_classes=num_classes,
                grid_size=grid_sizes[i],
                num_stacked_convs=num_stacked_convs,
                mask_kernel_channels=mask_kernel_channels,
                use_coordconv=True,  # use CoordConv in the head
                name=f'DynamicSOLOHead_{i}'
            )
            self.solo_heads.append(head)

        # Create the mask feature branch (used after the FPN outputs)
        self.mask_feat_branch = MaskFeatureFusion(out_channels=mask_kernel_channels)

    def call(self, inputs, training=False):
        # FPN outputs
        fpn_outputs = self.backbone(inputs, training=training)

        class_outputs = []
        mask_outputs = []

        # SOLO Head on each FPN scale
        for i, fpn_output in enumerate(fpn_outputs):
            class_out, mask_out = self.solo_heads[i](fpn_output, training=training)
            class_outputs.append(class_out)
            mask_outputs.append(mask_out)

        # Mask feature branch
        mask_feat = self.mask_feat_branch(fpn_outputs, training=training)

        return (class_outputs, mask_outputs, mask_feat)