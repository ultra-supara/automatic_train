"""
This code defines a YOLOv5 model with some variations on the backbone network architecture,
    like v3darknet, v4CSPdarknet53, v5CSPdarknet, v7backbone, and xCSPdarknet.
It consists of several custom-defined functions and classes for creating the model,
    along with some utility functions for creating and training the model.

Here's a brief description of the main components of the code:

1. **SiLU** class: A custom Keras layer that implements the Scaled Exponential Linear Unit (SiLU) activation function.

2. **DarknetConv2D** function: A wrapper function that applies certain initialization and regularization settings to a Conv2D layer.

3. **DarknetConv2D_BN_SiLU** function: A helper function that composes a Conv2D layer with batch normalization and SiLU activation function.

4. **Bottleneck**, **SPPBottleneck**, and **C3** functions: These functions are used to create various building blocks of the YOLOv5 model, like bottleneck layers, Spatial Pyramid Pooling layers, and CSPNet layers.

5. **yolov5_body** function: This is the main function for creating the YOLOv5 model with the specified input shape, anchor mask, number of classes, and backbone architecture.

6. **get_train_modelv5** function: This function takes the model created by the `yolov5_body` function, input shape, number of classes, anchors, and anchor masks, and creates a Keras model for training with custom loss function.
"""

from functools import wraps
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Concatenate, Input, BatchNormalization, Concatenate,
                                    Lambda, UpSampling2D, Conv2D, Layer, Add,
                                    MaxPooling2D, ZeroPadding2D)
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utilsv5.utils import compose

from yolomodels.nets.xCSPdarknet import darknet_bodyx
from yolomodels.nets.v3darknet import darknet_bodyv3
from yolomodels.nets.v4CSPdarknet53 import darknet_bodyv4
from yolomodels.nets.v5CSPdarknet import darknet_bodyv5
from yolomodels.nets.v7backbone import darknet_bodyv7

from yolomodels.yolo_training_v5 import yolo_loss
import os

class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

#---------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
# Convolution Block -> Convolution + Normalization + Activation Function
# DarknetConv2D + BatchNormalization + SiLU
#---------------------------------------------------#
def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())

def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.cv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    #---------------------------------------------------#
    #   The SPP structure, i.e., maximum pooling at different scales after stacking, is used.
    #---------------------------------------------------#
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    return x

def C3(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)
    #----------------------------------------------------------------#
    #   The trunk section loops over num_blocks, and inside the loop is the residual structure.
    #----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    #--------------------------------------------------------------------#
    #   Then a large residual edge shortconv is created, this large residual edge bypasses a lot of residual structure
    #--------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    #----------------------------------------------------------------#
    #   Stack the large residual edges back on top of each other
    #----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])

    #----------------------------------------------------------------#
    #   Final integration of the number of channels
    #----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.cv3')(route)

def yolov5_body(input_shape, anchors_mask, num_classes, backbone, alpha=1):
    depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    #dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

    base_channels       = int(0.67 * 64)  # 64
    base_depth          = max(round(0.75 * 3), 1)  # 3

    inputs      = Input(input_shape)

    if  backbone=="v3darknet":
        feat1,feat2,feat3 = darknet_bodyv3(inputs)

    elif backbone=="v4CSPdarknet53":
        feat1,feat2,feat3 = darknet_bodyv4(inputs)

    elif backbone=="v5CSPdarknet":
        feat1,feat2,feat3 = darknet_bodyv5(inputs)

    elif backbone=="v7backbone":
        feat1,feat2,feat3 = darknet_bodyv7(inputs)

    elif backbone=="xCSPdarknet":
        feat1,feat2,feat3 = darknet_bodyx(inputs)

    feat3 = SPPBottleneck(feat3, int(base_channels * 16), weight_decay=5e-4, name = 'sppfeat3')
    feat3 = C3(feat3, int(base_channels * 16), base_depth, shortcut=False, weight_decay=5e-4, name = 'feat3')

    P5          = DarknetConv2D_BN_SiLU(int(base_channels * 8), (1, 1), weight_decay=5e-4, name = 'conv_for_feat3')(feat3)
    P5_upsample = UpSampling2D()(P5)
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])
    P5_upsample = C3(P5_upsample, int(base_channels * 8), base_depth, shortcut = False, weight_decay=5e-4, name = 'conv3_for_upsample1')

    P4          = DarknetConv2D_BN_SiLU(int(base_channels * 4), (1, 1), weight_decay=5e-4, name = 'conv_for_feat2')(P5_upsample)
    P4_upsample = UpSampling2D()(P4)
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])
    P3_out      = C3(P4_upsample, int(base_channels * 4), base_depth, shortcut = False, weight_decay=5e-4, name = 'conv3_for_upsample2')

    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_SiLU(int(base_channels * 4), (3, 3), strides = (2, 2), weight_decay=5e-4, name = 'down_sample1')(P3_downsample)
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])
    P4_out          = C3(P3_downsample, int(base_channels * 8), base_depth, shortcut = False, weight_decay=5e-4, name = 'conv3_for_downsample1')

    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample   = DarknetConv2D_BN_SiLU(int(base_channels * 8), (3, 3), strides = (2, 2), weight_decay=5e-4, name = 'down_sample2')(P4_downsample)
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])
    P5_out          = C3(P4_downsample, int(base_channels * 16), base_depth, shortcut = False, weight_decay=5e-4, name = 'conv3_for_downsample2')

    out2 = DarknetConv2D(len(anchors_mask[2]) * (5 + num_classes), (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'yolo_head_P3')(P3_out)
    out1 = DarknetConv2D(len(anchors_mask[1]) * (5 + num_classes), (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'yolo_head_P4')(P4_out)
    out0 = DarknetConv2D(len(anchors_mask[0]) * (5 + num_classes), (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'yolo_head_P5')(P5_out)
    return Model(inputs, [out0, out1, out2])

def get_train_modelv5(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss,
        output_shape    = (1, ),
        name            = 'yolo_loss',
        arguments       = {
            'input_shape'       : input_shape,
            'anchors'           : anchors,
            'anchors_mask'      : anchors_mask,
            'num_classes'       : num_classes,
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 1 * (input_shape[0] * input_shape[1]) / (640 ** 2),
            'cls_ratio'         : 0.5 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
