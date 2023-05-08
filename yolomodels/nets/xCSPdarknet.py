"""
This code defines a custom neural network architecture using TensorFlow and Keras.
It's an implementation of the CSPDarknet with some modifications, such as the addition of a Focus layer and other custom layers like SiLU and SPPBottleneck.

1. **SiLU class**: A custom Keras layer that implements the Sigmoid Linear Unit (SiLU) activation function. The SiLU function is defined as f(x) = x * sigmoid(x).

    - `__init__`: Initializes the SiLU layer.
    - `call`: Applies the SiLU activation function to the input tensor.
    - `get_config`: Returns the configuration of the SiLU layer.
    - `compute_output_shape`: Returns the output shape of the layer given the input shape.

2. **Focus class**: A custom Keras layer that rearranges the input tensor by concatenating it in a specific pattern. The output tensor has 4 times the number of channels and half the spatial dimensions (width and height) of the input tensor.

    - `__init__`: Initializes the Focus layer.
    - `call`: Applies the focus operation on the input tensor.
    - `compute_output_shape`: Returns the output shape of the layer given the input shape.

3. **DarknetConv2D function**: A wrapper around the Keras Conv2D layer, which includes a custom kernel initializer (RandomNormal with stddev=0.02) and a kernel regularizer (l2).

4. **DarknetConv2D_BN_SiLU function**: Composes a DarknetConv2D layer followed by a BatchNormalization layer with momentum=0.97 and epsilon=0.001, and a SiLU activation layer.
    It creates a single convolution block in the modified CSPDarknet architecture.

5. **SPPBottleneck function**: Applies Spatial Pyramid Pooling (SPP) to the input tensor by performing max pooling with different kernel sizes (5x5, 9x9, and 13x13) and concatenating the results.
    The SPP operation captures multi-scale contextual information, which can be beneficial for object detection tasks.

6. **Bottleneck function**: Defines a bottleneck block that composes two DarknetConv2D_BN_SiLU layers with different kernel sizes (1x1 and 3x3) and optionally adds a residual connection (shortcut) from the input tensor.

7. **CSPLayer function**: Creates a Cross-Stage Partial Layer (CSP) that first splits the input tensor into two branches, applies bottleneck blocks to one branch, and then concatenates the two branches.
    The CSP layer is designed to improve the flow of gradients during training.

8. **resblock_body function**: Creates a sequence of layers composed of a ZeroPadding2D layer, a DarknetConv2D_BN_SiLU layer, and a CSPLayer.
    This sequence is applied multiple times in the modified CSPDarknet architecture.

9. **darknet_bodyx function**: Constructs the full modified CSPDarknet architecture by applying a Focus layer followed by a DarknetConv2D_BN_SiLU layer and then multiple instances of the resblock_body function.
    It returns three feature tensors (feat1, feat2, feat3) at different scales, suitable for multi-scale object detection tasks.

In summary, this code consists of two custom layers (SiLU and Focus), three utility functions (DarknetConv2D, DarknetConv2D_BN_SiLU, and SPPBottleneck)
    , and four main building blocks (Bottleneck, CSPLayer, resblock_body, and darknet_bodyx) that create the overall modified CSPDarknet
"""


from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                    Conv2D, Layer, MaxPooling2D,
                                    ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose


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

class Focus(Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x):
        return tf.concat(
            [x[...,  ::2,  ::2, :],
             x[..., 1::2,  ::2, :],
             x[...,  ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
             axis=-1
        )
#------------------------------------------------------#
# Single convolution DarknetConv2D
# Set your own padding method if the step size is 2.
#------------------------------------------------------#
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
#   convolution blocks -> convolution + standardization + activation function
#   DarknetConv2D + BatchNormalization + SiLU
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

def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    #---------------------------------------------------#
    # 使用されるSPP構造、すなわち、積層後に異なるスケールで最大限のプーリングを行う
    #---------------------------------------------------#
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    return x

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
    # トランク部はnum_blocksをループし、ループ内はresidual構造となっています。
    #----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    #--------------------------------------------------------------------#
    # 次に、大きな残差エッジであるshortconvを作成し、残差構造の多くを迂回させます。
    #--------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    #----------------------------------------------------------------#
    # 残っているエッジを再び積み重ねる
    #----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])

    #----------------------------------------------------------------#
    # チャンネル数の最終的な集約
    #----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.conv3')(route)

def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    #----------------------------------------------------------------#
    # ZeroPadding2D とステップサイズ 2x2 の畳み込みブロックによる縦横圧縮
    #----------------------------------------------------------------#

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')

#---------------------------------------------------#
# CSPdarknetの主要部分
# 入力は640x640x3の画像です
# 出力されるのは3つの有効な特徴層
#---------------------------------------------------#
def darknet_bodyx(x, weight_decay=5e-4):
    phi = 'l'
    depth_dict      = {'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict      = {'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
    base_channels   = int(wid_mul * 64)  # 64
    base_depth      = max(round(dep_mul * 3), 1)  # 3
    # 640, 640, 3 => 320, 320, 12
    x = Focus()(x)
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), weight_decay=weight_decay, name = 'backbone.backbone.stem.conv')(x)
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.backbone.dark2')
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark3')
    feat1 = x
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3
