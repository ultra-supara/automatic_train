"""
This code defines a YOLOv4 object detection model using TensorFlow and Keras.
The YOLOv4 model is a single-shot object detection model that is efficient and accurate.
The model is defined using various functions and layers in the code. Below is a high-level overview of the code:

1. Import necessary libraries and modules: The code imports the required libraries and modules for building the YOLOv4 model.

2. Define utility functions: The code defines several utility functions
    such as `relu6`, `hard_sigmoid`, `hard_swish`, `DarknetConv2D`, `DarknetConv2D_BN_Leaky`, and `_depthwise_conv_block`.
    These functions are used to build the YOLOv4 model architecture.

3. Define the `make_five_convs` function: This function performs five consecutive convolutional operations with depthwise separable convolutions.

4. Define the `yolov4_body` function: This function creates the main YOLOv4 model with a specified input shape, anchor masks, number of classes, backbone model, and alpha (a scaling factor for the number of filters in each layer).
    It builds the model by connecting various layers and returns a Keras `Model` object.

5. Define the `get_train_modelv4` function: This function wraps the YOLOv4 model created by `yolov4_body` with additional input layers for ground truth data (y_true) and a custom loss function (yolo_loss).
    This function returns a Keras `Model` object that can be used for training.

The code uses different backbones such as v3darknet, v4CSPdarknet53, v5CSPdarknet, v7backbone, and xCSPdarknet for the YOLOv4 model.
The choice of the backbone can be specified when calling the `yolov4_body` function.
"""

from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization, Multiply, Add,
                                    Concatenate, Conv2D, DepthwiseConv2D, LeakyReLU,
                                    Input, Lambda, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils.utilsv4.utils import compose

from yolomodels.nets.xCSPdarknet import darknet_bodyx
from yolomodels.nets.v3darknet import darknet_bodyv3
from yolomodels.nets.v4CSPdarknet53 import darknet_bodyv4
from yolomodels.nets.v5CSPdarknet import darknet_bodyv5
from yolomodels.nets.v7backbone import darknet_bodyv7

from yolomodels.yolo_training_v4 import yolo_loss
import os

def relu6(x):
    return K.relu(x, max_value=6)
def hard_sigmoid(x):
    return K.relu(x + 3.0, max_value=6.0) / 6.0
def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Activation(LeakyReLU(alpha=0.1))) #relu6  LeakyReLU(alpha=0.1)  hard_swish

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha = 1,
                            depth_multiplier=1, strides=(1, 1)):

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3), depthwise_initializer=RandomNormal(stddev=0.02),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False)(inputs)

    x = BatchNormalization()(x)
    x = Activation(relu6)(x) #relu6

    x = DarknetConv2D(pointwise_conv_filters, (1, 1),
                        padding='same',
                        use_bias=False,
                        strides=(1, 1))(x)
    x = BatchNormalization()(x)
    return Activation(relu6)(x) #relu6


def make_five_convs(input_x, num_filters):
    # 五次卷积
    y = DarknetConv2D_BN_Leaky(num_filters, (1,1))(input_x)
    y = _depthwise_conv_block(y, num_filters*2,alpha=1)
    y = DarknetConv2D_BN_Leaky(num_filters, (1,1))(y)
    y = _depthwise_conv_block(y, num_filters*2,alpha=1)
    y = DarknetConv2D_BN_Leaky(num_filters, (1,1))(y)
    return y


def yolov4_body(input_shape, anchors_mask, num_classes, backbone, alpha=1):
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

    # print('0000',feat1)
    # print('1111',feat2)
    # print('2222',feat3)

    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(feat3)
    P5 = _depthwise_conv_block(P5, int(1024* alpha))
    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(P5)

    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    P5_O = Concatenate()([maxpool1, maxpool2, maxpool3, P5])

    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(P5_O)
    P5 = _depthwise_conv_block(P5, int(1024* alpha))
    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(P5)

    P5_upsample = compose(DarknetConv2D_BN_Leaky(int(256* alpha), (1,1)), UpSampling2D(2))(P5)

    P4 = DarknetConv2D_BN_Leaky(int(256* alpha), (1,1))(feat2)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = make_five_convs(P4,int(256* alpha))

    P4_upsample = compose(DarknetConv2D_BN_Leaky(int(128* alpha), (1,1)), UpSampling2D(2))(P4)

    P3 = DarknetConv2D_BN_Leaky(int(128* alpha), (1,1))(feat1)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_five_convs(P3,int(128* alpha))

    P3_output = _depthwise_conv_block(P3, int(256* alpha))
    P3_output = DarknetConv2D(len(anchors_mask[0])*(num_classes+5), (1,1))(P3_output)

    P3_downsample = _depthwise_conv_block(P3, int(256* alpha), strides=(2,2))
    P4 = Concatenate()([P3_downsample, P4])

    P4 = make_five_convs(P4,int(256* alpha))

    P4_output = _depthwise_conv_block(P4, int(512* alpha))
    P4_output = DarknetConv2D(len(anchors_mask[1])*(num_classes+5), (1,1))(P4_output)

    P4_downsample = _depthwise_conv_block(P4, int(512* alpha), strides=(2,2))
    P5 = Concatenate()([P4_downsample, P5])

    P5 = make_five_convs(P5,int(512* alpha))

    P5_output = _depthwise_conv_block(P5, int(1024* alpha))

    P5_output = DarknetConv2D(len(anchors_mask[2])*(num_classes+5), (1,1))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])

def get_train_modelv4(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss,
        output_shape    = (1, ),
        name            = 'yolo_loss',
        arguments       = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask,
                            'num_classes' : num_classes}
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
