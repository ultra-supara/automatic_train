import os
from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                    Conv2D, Input, Lambda, Layer,
                                    UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils.utilsx.utils import compose

from yolomodels.nets.xCSPdarknet import darknet_bodyx
from yolomodels.nets.v3darknet import darknet_bodyv3
from yolomodels.nets.v4CSPdarknet53 import darknet_bodyv4
from yolomodels.nets.v5CSPdarknet import darknet_bodyv5
from yolomodels.nets.v7backbone import darknet_bodyv7

from yolomodels.yolo_training_x import get_yolo_loss


#---------------------------------------------------#
#   Construction of Panet network and obtaining prediction results
#---------------------------------------------------#
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
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.conv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)  # hidden channels
    #----------------------------------------------------------------#
    #   The trunk section loops over num_blocks, and inside the loop is the residual structure.
    #----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    #--------------------------------------------------------------------#
    #   Then create a large residual edge shortconv, this large residual edge bypasses a lot of residual structure
    #--------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    #----------------------------------------------------------------#
    #   Stack the large residual edges back on top of each other
    #----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])

    #----------------------------------------------------------------#
    #   Final integration of the number of channels
    #----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.conv3')(route)

def yolox_body(input_shape, num_classes, backbone, alpha=1):
    depth_dict      = {'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict      = {'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    depth, width    = 0.67, 0.75
    in_channels     = [256, 512, 1024]

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

    P5          = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (1, 1), weight_decay=5e-4, name = 'backbone.lateral_conv0')(feat3)  # 1024->512/32
    P5_upsample = UpSampling2D()(P5)  # 512/16
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])  # 512->1024/16
    P5_upsample = CSPLayer(P5_upsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=5e-4, name = 'backbone.C3_p4')  # 1024->512/16

    P4          = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (1, 1), weight_decay=5e-4, name = 'backbone.reduce_conv1')(P5_upsample)  # 512->256/16
    P4_upsample = UpSampling2D()(P4)  # 256/8
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])  # 256->512/8
    P3_out      = CSPLayer(P4_upsample, int(in_channels[0] * width), round(3 * depth), shortcut = False, weight_decay=5e-4, name = 'backbone.C3_p3')  # 1024->512/16

    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (3, 3), strides = (2, 2), weight_decay=5e-4, name = 'backbone.bu_conv2')(P3_downsample)  # 256->256/16
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])  # 256->512/16
    P4_out          = CSPLayer(P3_downsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=5e-4, name = 'backbone.C3_n3')  # 1024->512/16

    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (3, 3), strides = (2, 2), weight_decay=5e-4, name = 'backbone.bu_conv1')(P4_downsample)  # 256->256/16
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])  # 512->1024/32
    P5_out          = CSPLayer(P4_downsample, int(in_channels[2] * width), round(3 * depth), shortcut = False, weight_decay=5e-4, name = 'backbone.C3_n4')  # 1024->512/16

    fpn_outs    = [P3_out, P4_out, P5_out]
    yolo_outs   = []
    for i, out in enumerate(fpn_outs):
        # Channel number integration using 1x1 convolution
        stem    = DarknetConv2D_BN_SiLU(int(256 * width), (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'head.stems.' + str(i))(out)

        # Feature extraction using 3x3 convolution
        cls_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=5e-4, name = 'head.cls_convs.' + str(i) + '.0')(stem)
        cls_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=5e-4, name = 'head.cls_convs.' + str(i) + '.1')(cls_conv)
        #---------------------------------------------------#
        #   Determine the kind of feature point belongs to
        #   80, 80, num_classes
        #   40, 40, num_classes
        #   20, 20, num_classes
        #---------------------------------------------------#
        cls_pred = DarknetConv2D(num_classes, (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'head.cls_preds.' + str(i))(cls_conv)

        # Feature extraction using 3x3 convolution
        reg_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=5e-4, name = 'head.reg_convs.' + str(i) + '.0')(stem)
        reg_conv = DarknetConv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=5e-4, name = 'head.reg_convs.' + str(i) + '.1')(reg_conv)
        #---------------------------------------------------#
        #   Regression coefficient of characteristic points
        #   reg_pred 80, 80, 4
        #   reg_pred 40, 40, 4
        #   reg_pred 20, 20, 4
        #---------------------------------------------------#
        reg_pred = DarknetConv2D(4, (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'head.reg_preds.' + str(i))(reg_conv)
        #---------------------------------------------------#
        #   Determine whether the feature points have corresponding objects
        #   obj_pred 80, 80, 1
        #   obj_pred 40, 40, 1
        #   obj_pred 20, 20, 1
        #---------------------------------------------------#
        obj_pred = DarknetConv2D(1, (1, 1), strides = (1, 1), weight_decay=5e-4, name = 'head.obj_preds.' + str(i))(reg_conv)
        output   = Concatenate(axis = -1)([reg_pred, obj_pred, cls_pred])
        yolo_outs.append(output)
    return Model(inputs, yolo_outs)

def get_train_modelx(model_body, input_shape, num_classes):
    y_true = [Input(shape = (None, 5))]
    model_loss  = Lambda(
        get_yolo_loss(input_shape, len(model_body.output), num_classes),
        output_shape    = (1, ),
        name            = 'yolo_loss',
    )([*model_body.output, *y_true])

    model       = Model([model_body.input, *y_true], model_loss)
    return model
