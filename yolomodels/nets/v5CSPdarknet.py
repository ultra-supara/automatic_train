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

def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.cv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def C3(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)
    #----------------------------------------------------------------#
    # トランク部はnum_blocksをループし、ループ内はresidual構造
    #----------------------------------------------------------------#
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    #--------------------------------------------------------------------#
    #   次に、大きな残差エッジであるshortconvを作成し、残差構造を迂回させる
    #--------------------------------------------------------------------#
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    #----------------------------------------------------------------#
    #   残っている大きなエッジを再び積み重ねる
    #----------------------------------------------------------------#
    route = Concatenate()([x_1, x_2])
    #----------------------------------------------------------------#
    #   チャンネル数の最終整理
    #----------------------------------------------------------------#
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.cv3')(route)

def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    #---------------------------------------------------#
    #   異なるスケールで最大限のプーリングを行い、その後スタッキングを行うというSPP構造を採用
    #---------------------------------------------------#
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    return x

def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    #----------------------------------------------------------------#
    #   ZeroPadding2Dとステップサイズ2x2のコンボリューションブロックによる縦横圧縮
    #----------------------------------------------------------------#

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return C3(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')

#---------------------------------------------------#
# CSPdarknetの主要部分
# 入力は640x640x3の画像です
# 出力されるのは3つの有効な特徴層
#---------------------------------------------------#
def darknet_bodyv5(x, weight_decay=5e-4):
    # 640, 640, 3 => 320, 320, 12
    base_channels = 64
    base_depth = 3
    x = Focus()(x)
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), weight_decay=weight_decay, name = 'backbone.stem.conv')(x)
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.dark2')
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.dark3')
    feat1 = x
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3


"""
This code defines a CSPDarknet53 architecture, which is a popular backbone for object detection models.
CSPDarknet53 is a modification of the original Darknet53 with cross-stage hierarchical (CSH) connections, which improves information flow and gradient flow through the network.
This makes the network more efficient and improves the detection accuracy.

Here's a brief explanation of the main components in the code:

1. **SiLU**: The Sigmoid Linear Unit (SiLU) is an activation function defined as f(x) = x * sigmoid(x).
    It's also known as the Swish activation function. It has been shown to perform better than ReLU for deep networks in some cases.

2. **Focus**: This custom layer rearranges the input tensor by concatenating adjacent pixels to increase the channel depth.
    It effectively "focuses" the information in the image into a smaller spatial size with more channels, reducing the computational load for the subsequent layers.

3. **DarknetConv2D**: A custom wrapper around the Conv2D layer that applies specific initializers and regularizers to the weights.
    It also sets the padding depending on the stride.

4. **DarknetConv2D_BN_SiLU**: A composition of DarknetConv2D, BatchNormalization, and SiLU layers.
    It is a common block used in the CSPDarknet53 architecture.

5. **Bottleneck**: A bottleneck block that consists of two DarknetConv2D_BN_SiLU layers with different kernel sizes.
    It can include a residual connection (shortcut) if specified.

6. **C3**: A CSPDarknet block that combines a trunk part (composed of multiple Bottleneck blocks)
    and a large residual edge (shortconv), concatenated and followed by a final convolution layer.

7. **SPPBottleneck**: A Spatial Pyramid Pooling (SPP) bottleneck block that performs max-pooling at multiple scales and concatenates the results, followed by a convolution layer.

8. **resblock_body**: A function that creates a CSPDarknet block with specific parameters
    , followed by a downsampling step using ZeroPadding2D and DarknetConv2D_BN_SiLU layers with stride 2.

9. **darknet_bodyv5**: The main function that creates the CSPDarknet53 architecture by stacking the resblock_body functions with different parameters.
    It returns three feature maps (feat1, feat2, feat3) that can be used for object detection tasks.
"""
