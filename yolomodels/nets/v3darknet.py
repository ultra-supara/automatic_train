from functools import wraps

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, LeakyReLU, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose

#------------------------------------------------------#
#   Single convolution
#   DarknetConv2D
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
# convolution blocks -> convolution + standardization + activation function
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------------------------#
# 残差構造
# ZeroPadding2D とステップサイズ 2x2 の畳み込みブロックを用いた最初の縦横の圧縮
# その後、num_blockでループし、ループ内は残差構造です。
#---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, weight_decay=5e-4):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2), weight_decay=weight_decay)(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1), weight_decay=weight_decay)(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3), weight_decay=weight_decay)(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
# darknet53のメイン部分
# 入力は 416x416x3 の画像です
# 出力されるのは3つの有効な特徴層
#---------------------------------------------------#
def darknet_bodyv3(x, weight_decay=5e-4):
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3,3), weight_decay=weight_decay)(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3

"""
This code defines a modified version of the Darknet-53 architecture, which is a popular convolutional neural network (CNN) used as a backbone for object detection tasks.

The code defines several functions that build different parts of the Darknet-53 architecture:

1. `DarknetConv2D`: A wrapper function for the Conv2D layer. It sets the kernel initializer, padding, and regularization based on the given arguments.
2. `DarknetConv2D_BN_Leaky`: A function that composes a DarknetConv2D layer with Batch Normalization and Leaky ReLU activation function.
3. `resblock_body`: A function that builds a residual block.
    It takes the input tensor, number of filters, and number of blocks as arguments, and returns the output tensor after passing it through the residual blocks.
4. `darknet_bodyv3`: The main function that builds the Darknet-53 architecture.
    It takes the input tensor and returns three feature maps that can be used for object detection tasks.

This implementation differs slightly from the original Darknet-53 architecture
    , as it returns three feature maps instead of one, which is useful when using the model for object detection tasks with different scales.
"""
