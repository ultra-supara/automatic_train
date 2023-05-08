from functools import wraps

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                    Conv2D, Layer, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose


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
# convolution blocks -> convolution + standardization + activation function
#   DarknetConv2D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

#--------------------------------------------------------------------#
# CSPdarknetの構造化ブロック

# ZeroPadding2D とステップサイズ 2x2 のコンボリューションブロックを用いて縦横の圧縮を行う。
# その後、大きな残差エッジshortconvが作成され、残差構造の多くがバイパスされる
# バックボーン部分はnum_blocksをループし、その中に残差構造がある。
# CSPdarknetの構造ブロック全体では、1つの大きな残差ブロック＋内部に多数の小さな残差ブロックとなる。
#--------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, all_narrow=True, weight_decay=5e-4):
    #----------------------------------------------------------------#
    #   ZeroPadding2Dとステップサイズ2x2のコンボリューションブロックによる縦横圧縮
    #----------------------------------------------------------------#
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2), weight_decay=weight_decay)(preconv1)

    #--------------------------------------------------------------------#
    #   次に、残留構造の多くをバイパスする大きな残留エッジshortconvを作成します
    #--------------------------------------------------------------------#
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(preconv1)

    #----------------------------------------------------------------#
    #   トランク部はnum_blocksをループしており、ループ内には残差構造がある
    #----------------------------------------------------------------#
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1), weight_decay=weight_decay),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3), weight_decay=weight_decay))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(mainconv)

    #----------------------------------------------------------------#
    #   大きな残像のエッジを再び積み重ねる
    #----------------------------------------------------------------#
    route = Concatenate()([postconv, shortconv])

    # チャンネル数の最終統合
    return DarknetConv2D_BN_Mish(num_filters, (1,1), weight_decay=weight_decay)(route)

#---------------------------------------------------#
# CSPdarknet53の主要部分
# 入力は 416x416x3 の画像です
# 出力されるのは3つの有効な特徴層
#---------------------------------------------------#
def darknet_bodyv4(x, weight_decay=5e-4):
    x = DarknetConv2D_BN_Mish(32, (3,3), weight_decay=weight_decay)(x)
    x = resblock_body(x, 64, 1, False, weight_decay=weight_decay)
    x = resblock_body(x, 128, 2, weight_decay=weight_decay)
    x = resblock_body(x, 256, 8, weight_decay=weight_decay)
    feat1 = x
    x = resblock_body(x, 512, 8, weight_decay=weight_decay)
    feat2 = x
    x = resblock_body(x, 1024, 4, weight_decay=weight_decay)
    feat3 = x
    return feat1,feat2,feat3

"""
CSPDarknet-53アーキテクチャの修正版を定義しています。
オリジナルのDarknet-53アーキテクチャを改良し、CSH（Cross Stage Hierarchical）構造を追加することで、
勾配の流れを改善し、計算コストをほとんど増やさずにモデル容量を増加させることができる。

このコードでは、CSPDarknet-53アーキテクチャのさまざまな部分を構築するいくつかの関数とカスタムレイヤーを定義しています：

1. Mish : カスタム活性化関数層で、Mish活性化（`x * tanh(softplus(x))` と定義される）を計算する。
2. DarknetConv2D : Conv2D層のラッパー関数です。与えられた引数に基づいて、カーネルのイニシャライザー、パディング、正則化を設定する。
3. DarknetConv2D_BN_Mish : DarknetConv2D層にBatch NormalizationとMish活性化関数を組み合わせた関数です。
4. resblock_body : CSPDarknet-53アーキテクチャの残差ブロックをCSH構造で構築する関数です。入力テンソル、フィルタ数、ブロック数を引数にとり、残差ブロックを通過させた後の出力テンソルを返す。
5. darknet_bodyv4 : CSPDarknet-53アーキテクチャを構築するメイン関数である。入力テンソルを受け取り、物体検出タスクに使用可能な3つの特徴マップを返す。

CSPDarknet-53アーキテクチャは、オリジナルのDarknet-53アーキテクチャと比較して、勾配フローとモデル容量が改善されています。
この実装では、1つの特徴マップではなく3つの特徴マップを返すので、異なるスケールを持つ物体検出タスクにモデルを使用する場合に便利です。
"""
