"""
このコードでは、深層学習のためのKerasライブラリを使用して、YOLOv7モデルを定義します。
アーキテクチャは、いくつかのカスタムレイヤーと関数で構成されています。

Custom SiLUレイヤーを定義します :
このレイヤーは、SiLU（Sigmoid Linear Unit）活性化関数をその入力に適用します。

DarknetConv2D関数を定義します :
Conv2D層のラッパーで、YOLOv7のアーキテクチャに従ってkernel_initializerとkernel_regularizerをセットする。

DarknetConv2D_BN_SiLU関数を定義する :
DarknetConv2D、BatchNormalization、SiLUの各レイヤーを組み合わせたものです。

Transition_Block関数を定義します:
入力 'x' を受け取り、その高さと幅を圧縮し、畳み込みブロックを適用します。

Multi_Concat_Block関数を定義します:
入力「x」に複数の畳み込みレイヤーを適用し、その出力を連結させる。

SPPCSPC関数を定義する:
畳み込み、MAX-Pooling、連結の各レイヤーを組み合わせて入力 'x' に適用する。

fusion_rep_vgg関数を定義する:
畳み込み層の出力を推論モデルで結合させる。

RepConv関数を定義する:
畳み込み層の組み合わせを適用し、その出力を追加する。

yolov7_body関数を定義します:
上記で定義したカスタムレイヤーと関数を用いて、YOLOv7モデルを構築する。

このコードでは、これらのカスタムレイヤーと関数を使用して、与えられた入力形状、anchors_mask、num_classes、およびbackboneを持つYOLOv7モデルを作成する。
このモデルは物体検出タスク用に設計されており、学習させて画像に対する推論に使用することができます。
"""

import numpy as np
from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Layer, Conv2D, Input,
                                    ZeroPadding2D, Lambda, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model

from tensorflow.keras.regularizers import l2
from utils.utilsv7.utils import compose

from yolomodels.nets.xCSPdarknet import darknet_bodyx
from yolomodels.nets.v3darknet import darknet_bodyv3
from yolomodels.nets.v4CSPdarknet53 import darknet_bodyv4
from yolomodels.nets.v5CSPdarknet import darknet_bodyv5
from yolomodels.nets.v7backbone import darknet_bodyv7

from yolomodels.yolo_training_v7 import yolo_loss

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
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 0))}
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

def Transition_Block(x, c2, weight_decay=5e-4, name = ""):
    #----------------------------------------------------------------#
    #  ZeroPadding2D とstep size 2x2 の畳み込みブロックによる縦横圧縮
    #----------------------------------------------------------------#
    x_1 = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x_1 = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x_1)

    x_2 = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    x_2 = ZeroPadding2D(((1, 1),(1, 1)))(x_2)
    x_2 = DarknetConv2D_BN_SiLU(c2, (3, 3), strides=(2, 2), weight_decay=weight_decay, name = name + '.cv3')(x_2)
    y = Concatenate(axis=-1)([x_2, x_1])
    return y

def Multi_Concat_Block(x, c2, c3, n=4, e=1, ids=[0], weight_decay=5e-4, name = ""):
    c_ = int(c2 * e)

    x_1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    x_2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)

    x_all = [x_1, x_2]
    for i in range(n):
        x_2 = DarknetConv2D_BN_SiLU(c2, (3, 3), weight_decay=weight_decay, name = name + '.cv3.' + str(i))(x_2)
        x_all.append(x_2)
    y = Concatenate(axis=-1)([x_all[id] for id in ids])
    y = DarknetConv2D_BN_SiLU(c3, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(y)
    return y

def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), weight_decay=5e-4, name=""):
    c_ = int(2 * c2 * e)  # hidden channels
    x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    x1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv3')(x1)
    x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(x1)

    y1 = Concatenate(axis=-1)([x1] + [MaxPooling2D(pool_size=(m, m), strides=(1, 1), padding='same')(x1) for m in k])
    y1 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv5')(y1)
    y1 = DarknetConv2D_BN_SiLU(c_, (3, 3), weight_decay=weight_decay, name = name + '.cv6')(y1)

    y2 = DarknetConv2D_BN_SiLU(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    out = Concatenate(axis=-1)([y1, y2])
    out = DarknetConv2D_BN_SiLU(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv7')(out)

    return out

def fusion_rep_vgg(fuse_layers, trained_model, infer_model):
    for layer_name, use_bias, use_bn in fuse_layers:

        conv_kxk_weights = trained_model.get_layer(layer_name + '.rbr_dense.0').get_weights()[0]
        conv_1x1_weights = trained_model.get_layer(layer_name + '.rbr_1x1.0').get_weights()[0]

        if use_bias:
            conv_kxk_bias = trained_model.get_layer(layer_name + '.rbr_dense.0').get_weights()[1]
            conv_1x1_bias = trained_model.get_layer(layer_name + '.rbr_1x1.0').get_weights()[1]
        else:
            conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))
            conv_1x1_bias = np.zeros((conv_1x1_weights.shape[-1],))

        if use_bn:
            gammas_kxk, betas_kxk, means_kxk, var_kxk = trained_model.get_layer(layer_name + '.rbr_dense.1').get_weights()
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = trained_model.get_layer(layer_name + '.rbr_1x1.1').get_weights()

        else:
            gammas_1x1, betas_1x1, means_1x1, var_1x1 = [np.ones((conv_1x1_weights.shape[-1],)),
                                                        np.zeros((conv_1x1_weights.shape[-1],)),
                                                        np.zeros((conv_1x1_weights.shape[-1],)),
                                                        np.ones((conv_1x1_weights.shape[-1],))]
            gammas_kxk, betas_kxk, means_kxk, var_kxk = [np.ones((conv_kxk_weights.shape[-1],)),
                                                        np.zeros((conv_kxk_weights.shape[-1],)),
                                                        np.zeros((conv_kxk_weights.shape[-1],)),
                                                        np.ones((conv_kxk_weights.shape[-1],))]
        gammas_res, betas_res, means_res, var_res = [np.ones((conv_1x1_weights.shape[-1],)),
                                                    np.zeros((conv_1x1_weights.shape[-1],)),
                                                    np.zeros((conv_1x1_weights.shape[-1],)),
                                                    np.ones((conv_1x1_weights.shape[-1],))]

        # _fuse_bn_tensor(self.rbr_dense)
        w_kxk = (gammas_kxk / np.sqrt(np.add(var_kxk, 1e-3))) * conv_kxk_weights
        b_kxk = (((conv_kxk_bias - means_kxk) * gammas_kxk) / np.sqrt(np.add(var_kxk, 1e-3))) + betas_kxk

        # _fuse_bn_tensor(self.rbr_dense)
        kernel_size = w_kxk.shape[0]
        in_channels = w_kxk.shape[2]
        w_1x1 = np.zeros_like(w_kxk)
        w_1x1[kernel_size // 2, kernel_size // 2, :, :] = (gammas_1x1 / np.sqrt(np.add(var_1x1, 1e-3))) * conv_1x1_weights
        b_1x1 = (((conv_1x1_bias - means_1x1) * gammas_1x1) / np.sqrt(np.add(var_1x1, 1e-3))) + betas_1x1

        w_res = np.zeros_like(w_kxk)
        for i in range(in_channels):
            w_res[kernel_size // 2, kernel_size // 2, i % in_channels, i] = 1
        w_res = ((gammas_res / np.sqrt(np.add(var_res, 1e-3))) * w_res)
        b_res = (((0 - means_res) * gammas_res) / np.sqrt(np.add(var_res, 1e-3))) + betas_res

        weight = [w_res, w_1x1, w_kxk]
        bias = [b_res, b_1x1, b_kxk]

        infer_model.get_layer(layer_name).set_weights([np.array(weight).sum(axis=0), np.array(bias).sum(axis=0)])

def RepConv(x, c2, mode="train", weight_decay=5e-4, name=""):
    if mode == "predict":
        out = DarknetConv2D(c2, (3, 3), name = name, use_bias=True, weight_decay=weight_decay, padding='same')(x)
        out = SiLU()(out)
    elif mode == "train":
        x1 = DarknetConv2D(c2, (3, 3), name = name + '.rbr_dense.0', use_bias=False, weight_decay=weight_decay, padding='same')(x)
        x1 = BatchNormalization(momentum = 0.97, epsilon = 0.001, name = name + '.rbr_dense.1')(x1)
        x2 = DarknetConv2D(c2, (1, 1), name = name + '.rbr_1x1.0', use_bias=False, weight_decay=weight_decay, padding='same')(x)
        x2 = BatchNormalization(momentum = 0.97, epsilon = 0.001, name = name + '.rbr_1x1.1')(x2)

        out = Add()([x1, x2])
        out = SiLU()(out)
    return out
#---------------------------------------------------#
"""
`yolov7_body` function:

This function builds the main body of the YOLOv7 model. It takes the following arguments:
1. input_shape: the shape of the input tensor, e.g. (416, 416, 3) for an RGB image.
2. anchors_mask: the anchor masks used for generating anchor boxes.
3. num_classes: the number of classes in the dataset.
4. backbone: the name of the backbone model to use, e.g., "v3darknet", "v4CSPdarknet53", etc.
5. alpha (optional): a coefficient to scale the model size (default is 1).

The function first initializes some architecture-specific parameters based on the `phi` variable
    , which is hardcoded to 'l' (for the 'large' version of the model).
    Then, it builds the backbone model, which is responsible for extracting high-level features from the input image.
    The backbone is chosen based on the `backbone` argument.

After that, the function constructs the PANet (Path Aggregation Network) and the feature pyramid
    , which are responsible for object detection at different scales.
    The PANet consists of several Multi_Concat_Block and Transition_Block layers
    , which help in aggregating features from different levels of the backbone.

Finally, the function adds the output layers (called 'yolo_head') for each scale.
Each output layer has a depth of `len(anchors_mask[l]) * (5 + num_classes)`, where `l` is the level of the feature pyramid.
The output layers are responsible for predicting the bounding box coordinates, objectness score, and class probabilities.
"""
#---------------------------------------------------#
def yolov7_body(input_shape, anchors_mask, num_classes, backbone, alpha=1):
    #-----------------------------------------------#
    #   Defined parameters for different yolov7 versions
    #-----------------------------------------------#
    phi='l'
    mode="train"
    transition_channels = {'l' : 32, 'x' : 40}[phi]
    block_channels      = 32
    panet_channels      = {'l' : 32, 'x' : 64}[phi]
    e       = {'l' : 2, 'x' : 1}[phi]
    n       = {'l' : 4, 'x' : 6}[phi]
    ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]

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

    P5          = SPPCSPC(feat3, transition_channels * 16, weight_decay=5e-4, name="sppcspc")
    P5_conv     = DarknetConv2D_BN_SiLU(transition_channels * 8, (1, 1), weight_decay=5e-4, name="conv_for_P5")(P5)
    P5_upsample = UpSampling2D()(P5_conv)
    P4          = Concatenate(axis=-1)([DarknetConv2D_BN_SiLU(transition_channels * 8, (1, 1), weight_decay=5e-4, name="conv_for_feat2")(feat2), P5_upsample])
    P4          = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, weight_decay=5e-4, name="conv3_for_upsample1")

    P4_conv     = DarknetConv2D_BN_SiLU(transition_channels * 4, (1, 1), weight_decay=5e-4, name="conv_for_P4")(P4)
    P4_upsample = UpSampling2D()(P4_conv)
    P3          = Concatenate(axis=-1)([DarknetConv2D_BN_SiLU(transition_channels * 4, (1, 1), weight_decay=5e-4, name="conv_for_feat1")(feat1), P4_upsample])
    P3          = Multi_Concat_Block(P3, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids, weight_decay=5e-4, name="conv3_for_upsample2")

    P3_downsample = Transition_Block(P3, transition_channels * 4, weight_decay=5e-4, name="down_sample1")
    P4 = Concatenate(axis=-1)([P3_downsample, P4])
    P4 = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, weight_decay=5e-4, name="conv3_for_downsample1")

    P4_downsample = Transition_Block(P4, transition_channels * 8, weight_decay=5e-4, name="down_sample2")
    P5 = Concatenate(axis=-1)([P4_downsample, P5])
    P5 = Multi_Concat_Block(P5, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids,weight_decay=5e-4, name="conv3_for_downsample2")

    if phi == "l":
        P3 = RepConv(P3, transition_channels * 8, mode, weight_decay=5e-4, name="rep_conv_1")
        P4 = RepConv(P4, transition_channels * 16, mode, weight_decay=5e-4, name="rep_conv_2")
        P5 = RepConv(P5, transition_channels * 32, mode, weight_decay=5e-4, name="rep_conv_3")
    else:
        P3 = DarknetConv2D_BN_SiLU(transition_channels * 8, (3, 3), strides=(1, 1), weight_decay=5e-4, name="rep_conv_1")(P3)
        P4 = DarknetConv2D_BN_SiLU(transition_channels * 16, (3, 3), strides=(1, 1), weight_decay=5e-4, name="rep_conv_2")(P4)
        P5 = DarknetConv2D_BN_SiLU(transition_channels * 32, (3, 3), strides=(1, 1), weight_decay=5e-4, name="rep_conv_3")(P5)

    # len(anchors_mask[2]) = 3
    # 5 + num_classes -> 4 + 1 + num_classes
    # 4 is the regression coefficient of the a priori box, 1 is the sigmoid to fix the value to 0-1
    # , and num_classes is used to determine what class of objects the a priori box is
    # bs, 20, 20, 3 * (4 + 1 + num_classes)
    out2 = DarknetConv2D(len(anchors_mask[2]) * (5 + num_classes), (1, 1), weight_decay=5e-4, strides = (1, 1), name = 'yolo_head_P3')(P3)
    out1 = DarknetConv2D(len(anchors_mask[1]) * (5 + num_classes), (1, 1), weight_decay=5e-4, strides = (1, 1), name = 'yolo_head_P4')(P4)
    out0 = DarknetConv2D(len(anchors_mask[0]) * (5 + num_classes), (1, 1), weight_decay=5e-4, strides = (1, 1), name = 'yolo_head_P5')(P5)
    return Model(inputs, [out0, out1, out2])

"""
get_train_modelv7 function:
This function creates a training model that wraps the YOLOv7 model body and the loss function.
It takes the following arguments:
model_body: the main body of the YOLOv7 model, as created by the yolov7_body function.
input_shape: the shape of the input tensor, e.g. (416, 416, 3) for an RGB image.
num_classes: the number of classes in the dataset.
anchors: the anchor boxes used for generating the bounding boxes.
anchors_mask: the anchor masks used for generating anchor boxes.

The function first creates input tensors for the ground truth data (y_true)
    , which are used for computing the loss during training.
Then, it creates a Lambda layer that computes the YOLOv7 loss using the yolo_loss function.
The loss function takes several arguments, including the input shape, anchors, anchor masks, number of classes,
    and some hyperparameters for balancing the loss components.
"""

def get_train_modelv7(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), 2)) for l in range(len(anchors_mask))] + [Input(shape = [None, 5])]
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
