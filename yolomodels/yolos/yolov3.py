"""
The code is a Python script that defines a YOLOv3 object detection model using TensorFlow and Keras.
The script also includes helper functions to build different backbone architectures and get the training model for YOLOv3.

Here's a summary of the main components of the code:

1. `DarknetConv2D`: A wrapper function around the Keras `Conv2D` layer with custom initializers and regularization.

2. `DarknetConv2D_BN_Leaky`: A function that composes a `DarknetConv2D` layer, followed by a `BatchNormalization` layer and a `LeakyReLU` activation function.

3. `make_five_conv`: A function that creates a series of five convolutional layers with alternating 1x1 and 3x3 filters.

4. `make_yolo_head`: A function that creates the YOLO detection head, which consists of a 3x3 convolutional layer followed by a 1x1 convolutional layer.

5. `yolov3_body`: A function that creates the YOLOv3 architecture with a specified backbone. This function takes `input_shape`, `anchors_mask`, `num_classes`, and `backbone` as arguments, and returns a Keras `Model`.

6. `get_train_modelv3`: A function that returns a training model for YOLOv3, given the model body, input shape, number of classes, anchors, and anchor masks.

The script imports various functions and classes from TensorFlow and Keras
, and also imports backbone architectures from other modules (e.g., `v3darknet`, `v4CSPdarknet53`, `v5CSPdarknet`, `v7backbone`, and `xCSPdarknet`)

This code is useful for training and implementing YOLOv3 object detection models. You can use the `yolov3_body` function to create a YOLOv3 model with a specified backbone, and the `get_train_modelv3` function to obtain the training model to be used for training on your dataset.
"""

from functools import wraps
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Concatenate, Input, Lambda, UpSampling2D,Conv2D,
                                    BatchNormalization, Activation, LeakyReLU)
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2
from utils.utilsv3.utils import compose

from yolomodels.nets.xCSPdarknet import darknet_bodyx
from yolomodels.nets.v3darknet import darknet_bodyv3
from yolomodels.nets.v4CSPdarknet53 import darknet_bodyv4
from yolomodels.nets.v5CSPdarknet import darknet_bodyv5
from yolomodels.nets.v7backbone import darknet_bodyv7


from yolomodels.yolo_training_v3 import yolo_loss

import os


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
# DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def make_five_conv(x, num_filters, weight_decay=5e-4):
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=5e-4)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=5e-4)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=5e-4)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=5e-4)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=5e-4)(x)
    return x

def make_yolo_head(x, num_filters, out_filters, weight_decay=5e-4):
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=5e-4)(x)
    # 255->3, 85->3, 4 + 1 + 80
    y = DarknetConv2D(out_filters, (1,1), weight_decay=5e-4)(y)
    return y

#---------------------------------------------------#
#   Construction of FPN network and obtaining prediction results
#---------------------------------------------------#
def yolov3_body(input_shape, anchors_mask, num_classes, backbone, alpha=1):

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

    #---------------------------------------------------#
    #   First feature layer
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x   = make_five_conv(feat3, 512)
    P5  = make_yolo_head(x, 512, len(anchors_mask[0]) * (num_classes+5))

    # 13,13,512 -> 13,13,256 -> 26,26,256
    x   = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(x)

    # 26,26,256 + 26,26,512 -> 26,26,768
    x   = Concatenate()([x, feat2])
    #---------------------------------------------------#
    #   Second feature layer
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x   = make_five_conv(x, 256)
    P4  = make_yolo_head(x, 256, len(anchors_mask[1]) * (num_classes+5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x   = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x   = Concatenate()([x, feat1])
    #---------------------------------------------------#
    #   Third feature layer
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x   = make_five_conv(x, 128)
    P3  = make_yolo_head(x, 128, len(anchors_mask[2]) * (num_classes+5))
    return Model(inputs, [P5, P4, P3])


def get_train_modelv3(model_body, input_shape, num_classes, anchors, anchors_mask):
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
            'obj_ratio'         : 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
            'cls_ratio'         : 1 * (num_classes / 80),
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
