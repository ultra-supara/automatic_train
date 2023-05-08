"""
This code defines the architecture and training model for the YOLOv4 object detection algorithm using various backbone networks
such as MobileNetV1, MobileNetV2, MobileNetV3, GhostNet, DenseNet, VGG16, ResNet50, InceptionV3, Xception, and CSPDarknet53.
The YOLOv4 model can be trained using the `get_train_model` function with the appropriate input parameters.

Here's a brief explanation of each part of the code:

1. Imports necessary libraries and utility functions.
2. Defines activation functions like relu6, hard_sigmoid, and hard_swish.
3. Wraps the Conv2D layer with default parameters for YOLOv4 in the `DarknetConv2D` function.
4. Defines the `DarknetConv2D_BN_Leaky` function to create a sequence of DarknetConv2D, BatchNormalization, and LeakyReLU layers.
5. Implements the `_depthwise_conv_block` function, which defines a depthwise separable convolution block with BatchNormalization and ReLU6 activation.
6. The `make_five_convs` function is a helper function that creates five consecutive convolution layers with the specified number of filters.
7. Defines the `yolov4_body` function, which creates the YOLOv4 model using the specified backbone network.
8. The `get_train_model` function creates a training model for YOLOv4, including the YOLOv4 model and the loss function.

You can train the YOLOv4 model on your dataset using the `get_train_model` function, providing the appropriate input shape, number of classes, anchors, and anchors_mask.
Once the model is trained, you can use it for object detection tasks.
"""

from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization, Multiply, Add,
                                    Concatenate, Conv2D, DepthwiseConv2D, LeakyReLU,
                                    Input, Lambda, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils.utils import compose
from yolomodels.attention import cbam_block, eca_block, se_block
from yolomodels.nets.densenet121 import DenseNet
from yolomodels.nets.ghostnet import Ghostnet
from yolomodels.nets.mobilenet_v1 import MobileNetV1
from yolomodels.nets.mobilenet_v2 import MobileNetV2
from yolomodels.nets.mobilenet_v3 import MobileNetV3
from yolomodels.nets.CSPdarknet53 import darknet_body
from yolomodels.nets.vgg16 import VGG16
from yolomodels.nets.resnet50 import ResNet50
from yolomodels.nets.Inception_v3 import InceptionV3
from yolomodels.nets.Xception import Xception
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

dir_nets_path = "yolos/nets"
nets_names = os.listdir(dir_nets_path)

def yolov4_body(input_shape, anchors_mask, num_classes, backbone, alpha=1):
    inputs      = Input(input_shape)
    print(backbone)
    #---------------------------------------------------#
    #   mobilnetのバックボーンモデルを生成し、3つの有効な機能レイヤーを取得する
    #---------------------------------------------------#
    if backbone=="mobilenet_v1":
        #---------------------------------------------------#
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        feat1,feat2,feat3 = MobileNetV1(inputs, alpha=alpha)
    elif backbone=="mobilenet_v2":
        #---------------------------------------------------#
        #   52,52,32；26,26,92；13,13,320
        #---------------------------------------------------#
        feat1,feat2,feat3 = MobileNetV2(inputs, alpha=alpha)
    elif backbone=="mobilenet_v3":
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = MobileNetV3(inputs, alpha=alpha)
    elif backbone=="ghostnet":
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = Ghostnet(inputs)
    elif backbone=="darknet":
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = darknet_body(inputs)
    elif backbone=="vgg16":
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = VGG16(inputs)
    elif backbone=="resnet50":
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = ResNet50(inputs)
    elif backbone in ["densenet121", "densenet169", "densenet201"]:
    # elif backbone=="densenet":
        #---------------------------------------------------#
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        feat1,feat2,feat3 = DenseNet(inputs, backbone)
    elif backbone=="Inception_v3":
        feat1,feat2,feat3 = InceptionV3(inputs)
    elif backbone=="Xception":
        feat1,feat2,feat3 = Xception(inputs)
    elif backbone=="CSPdarknet53":
        feat1,feat2,feat3 = darknet_body(inputs)

    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenetv1, mobilenetv2, mobilenetv3, ghostnet, densenet121, densenet169, densenet201.'.format(backbone))

    print('0000',feat1)
    print('1111',feat2)
    print('2222',feat3)

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

def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
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
