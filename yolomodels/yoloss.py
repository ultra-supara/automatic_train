"""
This code defines a YOLOv4 model implementation in TensorFlow.
It contains functions to create the YOLOv4 network architecture, as well as a function to create the training model by incorporating the YOLOv4 loss function.

The code first imports the required libraries and functions.
Then, it defines several helper functions, such as `Mish` activation, `DarknetConv2D_BN_Leaky`, `make_five_convs`, and the main `yolo_body` function to create the YOLOv4 architecture.

The `yolo_body` function takes in the input shape, anchor masks, and the number of classes as arguments and creates the YOLOv4 network architecture by connecting the different layers and feature maps.
It returns a TensorFlow `Model` object containing the network architecture.

The `get_train_model` function creates the training model by incorporating the YOLOv4 loss function, which is specified in the `Lambda` layer.
It takes in arguments like model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing, focal_loss, alpha, and gamma.
It returns a TensorFlow `Model` object with the complete training model including the loss function.

This implementation uses a custom `Mish` activation function and incorporates the `Ghostnet` backbone in the architecture.
"""

from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Input, Layer,
                                    Lambda, LeakyReLU, MaxPooling2D, ReLU, Add, Conv2D,
                                    UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model
from utils.utils import compose

#from nets.CSPdarknet53 import DarknetConv2D, resblock_body, darknet_body
#from yolo.nets.gghost import DarknetConv2D, Ghostnet, _ghost_bottleneck
from yolomodels.yolo_training_v4 import yolo_loss

class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
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
        # Mish())

#---------------------------------------------------#
#   Perform five convolutions
#---------------------------------------------------#
def make_five_convs(x, num_filters, weight_decay=5e-4):
    # 五次卷积
    # inputs = Conv2D(num_filters, (1, 1), strides=(1, 1), use_bias=False)(x)
    # inputs = BatchNormalization()(inputs)
    # inputs = ReLU()(inputs)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    # x = Add()([x,inputs])
    # x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    # x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    return x

#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, weight_decay=5e-4):
    input_shape = [416,416,3]
    inputs      = Input(input_shape)
    #---------------------------------------------------#
    # CSPdarknet53 のバックボーンモデルを生成する。
    # として、それぞれの形状を持つ3つの有効な特徴レイヤーを取得する：
    #   52,52,256
    #   26,26,512
    #   13,13,1024
    #---------------------------------------------------#
    # feat1,feat2,feat3 = darknet_body(inputs,weight_decay=weight_decay)
    feat1,feat2,feat3 = Ghostnet(inputs)

    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3), weight_decay=weight_decay)(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(P5)

    # The SPP structure, i.e., maximum pooling at different scales after stacking, is used.
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    # maxpool1 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    # maxpool2 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(maxpool1)
    # maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(maxpool2)

    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3), weight_decay=weight_decay)(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(P5)

    # 13,13,512 -> 13,13,256 -> 26,26,256
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1), weight_decay=weight_decay), UpSampling2D(2))(P5)
    # 26,26,512 -> 26,26,256
    P4 = DarknetConv2D_BN_Leaky(256, (1,1), weight_decay=weight_decay)(feat2)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P4, P5_upsample])

    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256, weight_decay=weight_decay)

    # 26,26,256 -> 26,26,128 -> 52,52,128
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1), weight_decay=weight_decay), UpSampling2D(2))(P4)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(128, (1,1), weight_decay=weight_decay)(feat1)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3 = Concatenate()([P3, P4_upsample])

    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3 = make_five_convs(P3, 128, weight_decay=weight_decay)

    #---------------------------------------------------#
    #   The third feature layer
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    P3_output = DarknetConv2D_BN_Leaky(256, (3,3), weight_decay=weight_decay)(P3)
    P3_output = DarknetConv2D(len(anchors_mask[0])*(num_classes+5), (1,1), weight_decay=weight_decay)(P3_output)

    # 52,52,128 -> 26,26,256
    P3_downsample = ZeroPadding2D(((1,0),(1,0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2), weight_decay=weight_decay)(P3_downsample)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P3_downsample, P4])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256, weight_decay=weight_decay)

    #---------------------------------------------------#
    #   Second feature layer
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    P4_output = DarknetConv2D_BN_Leaky(512, (3,3), weight_decay=weight_decay)(P4)
    P4_output = DarknetConv2D(len(anchors_mask[1])*(num_classes+5), (1,1), weight_decay=weight_decay)(P4_output)

    # 26,26,256 -> 13,13,512
    P4_downsample = ZeroPadding2D(((1,0),(1,0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2), weight_decay=weight_decay)(P4_downsample)
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P5 = Concatenate()([P4_downsample, P5])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = make_five_convs(P5, 512, weight_decay=weight_decay)

    #---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    P5_output = DarknetConv2D_BN_Leaky(1024, (3,3), weight_decay=weight_decay)(P5)
    P5_output = DarknetConv2D(len(anchors_mask[2])*(num_classes+5), (1,1), weight_decay=weight_decay)(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])

# 損失関数
def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing, focal_loss, alpha, gamma):
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
            'label_smoothing'   : label_smoothing,
            'focal_loss'        : focal_loss,
            'focal_loss_ratio'  : 10,
            'alpha'             : alpha,
            'gamma'             : gamma,
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
