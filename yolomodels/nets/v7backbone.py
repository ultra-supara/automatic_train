"""
CSPDarknet53アーキテクチャの修正版であるCSPDarknetV7アーキテクチャのKeras実装を定義
より良いgradient flowと改良された特徴量の結合のために設計しなおした
"""

from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Layer, MaxPooling2D, ZeroPadding2D)
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

#------------------------------------------------------#
# Single convolution DarknetConv2D
# Set your own padding method if the step size is 2.
#------------------------------------------------------#
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

def Transition_Block(x, c2, weight_decay=5e-4, name = ""):
    #----------------------------------------------------------------#
    #   ZeroPadding2Dとステップサイズ2x2のコンボリューションブロックによる縦横圧縮
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

#---------------------------------------------------#
# CSPdarknetの主要部分
# 入力は640x640x3の画像です
# 出力されるのは3つの有効な特徴層
#---------------------------------------------------#
def darknet_bodyv7(x,  weight_decay=5e-4):
    #-----------------------------------------------#
    #   入力画像は640, 640, 3
    #-----------------------------------------------#
    phi = 'l'
    transition_channels = {'l' : 32, 'x' : 40}[phi]
    block_channels = 32
    n = {'l' : 4, 'x' : 6}[phi]
    ids = {
        'l' : [-1, -3, -5, -6],
        'x' : [-1, -3, -5, -7, -8],
    }[phi]
    #---------------------------------------------------#
    #   base_channels  : 初期値64
    #---------------------------------------------------#
    # 320, 320, 3 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(transition_channels, (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'backbone.stem.0')(x)
    x = ZeroPadding2D(((1, 1),(1, 1)))(x)
    x = DarknetConv2D_BN_SiLU(transition_channels * 2, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.stem.1')(x)
    x = DarknetConv2D_BN_SiLU(transition_channels * 2, (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'backbone.stem.2')(x)

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 1),(1, 1)))(x)
    x = DarknetConv2D_BN_SiLU(transition_channels * 4, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.dark2.0')(x)
    x = Multi_Concat_Block(x, block_channels * 2, transition_channels * 8, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark2.1')

    # 160, 160, 128 => 80, 80, 256
    x = Transition_Block(x, transition_channels * 4, weight_decay=weight_decay, name = 'backbone.dark3.0')
    x = Multi_Concat_Block(x, block_channels * 4, transition_channels * 16, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark3.1')
    feat1 = x

    # 80, 80, 256 => 40, 40, 512
    x = Transition_Block(x, transition_channels * 8, weight_decay=weight_decay, name = 'backbone.dark4.0')
    x = Multi_Concat_Block(x, block_channels * 8, transition_channels * 32, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark4.1')
    feat2 = x

    # 40, 40, 512 => 20, 20, 1024
    x = Transition_Block(x, transition_channels * 16, weight_decay=weight_decay, name = 'backbone.dark5.0')
    x = Multi_Concat_Block(x, block_channels * 8, transition_channels * 32, n=n, ids=ids, weight_decay=weight_decay, name = 'backbone.dark5.1')
    feat3 = x
    return feat1, feat2, feat3

"""
Here's a detailed breakdown of the code by class and function:

1. **SiLU class**: A custom Keras layer that implements the Sigmoid Linear Unit (SiLU) activation function. The SiLU function is defined as f(x) = x * sigmoid(x).

    - `__init__`: Initializes the SiLU layer.
    - `call`: Applies the SiLU activation function to the input tensor.
    - `get_config`: Returns the configuration of the SiLU layer.
    - `compute_output_shape`: Returns the output shape of the layer given the input shape.

2. **DarknetConv2D function**: A wrapper around the Keras Conv2D layer, which includes a custom kernel initializer (RandomNormal with stddev=0.02) and a kernel regularizer (l2).

3. **DarknetConv2D_BN_SiLU function**: Composes a DarknetConv2D layer followed by a BatchNormalization layer with momentum=0.97 and epsilon=0.001, and a SiLU activation layer.
    It creates a single convolution block in the CSPDarknet architecture.

4. **Transition_Block function**: Takes an input tensor (x), the number of output channels (c2), an optional weight decay (default 5e-4), and a name for the block.
    It downsamples the input tensor using max pooling, and concatenates the downsampled tensor with another tensor obtained by applying a 3x3 convolution with stride 2.
    This block is a key component of the CSPDarknet architecture.

5. **Multi_Concat_Block function**: Takes an input tensor (x),
                                    the number of output channels for intermediate convolutions (c2) and
                                    final 1x1 convolution (c3),
                                    the number of 3x3 convolutions to create (n),
                                    an expansion factor for intermediate channels (e),
                                    a list of indices to select tensors for concatenation (ids),
                                    an optional weight decay (default 5e-4), and a name for the block.
    It creates a set of convolutional layers and concatenates their outputs based on the given indices.
    This block is responsible for the CSPDarknet's characteristic cross-stage hierarchy.

6. **darknet_bodyv7 function**: Takes an input tensor (x) and an optional weight decay (default 5e-4).
    It creates the full CSPDarknetV7 architecture by sequentially applying a stem, followed by multiple Multi_Concat_Block and Transition_Block instances.
    It returns three feature tensors (feat1, feat2, feat3) at different scales, suitable for multi-scale object detection tasks.

In summary, this code consists of a custom activation layer (SiLU), two utility functions (DarknetConv2D and DarknetConv2D_BN_SiLU) that simplify the creation of convolution blocks
    , and three main building blocks (Transition_Block, Multi_Concat_Block, and darknet_bodyv7) that create the overall CSPDarknetV7 architecture.
"""
