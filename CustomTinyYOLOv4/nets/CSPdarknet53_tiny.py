from functools import wraps

import tensorflow as tf
from keras.initializers import random_normal
from keras.layers import (Concatenate, Conv2D, Lambda, MaxPooling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

#------------------------------------------------------#
# Darknet single convolution conv2 d
# If the step size is 2, set the fill method yourself.
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
    
#---------------------------------------------------#
# Convolution block
# DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

'''
                    input
                      |
            DarknetConv2D_BN_Leaky
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
            DarknetConv2D_BN_Leaky          |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1    DarknetConv2D_BN_Leaky          |
    |                 |                     |
    -------------Concatenate                |
                      |                     |
        ----DarknetConv2D_BN_Leaky          |
        |             |                     |
      feat       Concatenate-----------------
                      |
                 MaxPooling2D
'''
#---------------------------------------------------#
# Structural blocks of Cs pdarknet tiny
# There is a large residual margin
# This large residual edge bypasses many residual structures
#---------------------------------------------------#
def resblock_body(x, num_filters):
    # Use a 3x3 convolution for feature integration
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    # lead to a large residual marginal path
    route = x

    # The channels of the feature layer are segmented and the second part is treated as the backbone part.
    x = Lambda(route_group, arguments={'groups':2, 'group_id':1})(x) 
    # 3x3 convolution on the dorsal side
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    # leads to a small residual marginal path 1
    route_1 = x
    # 3x3 convolution on the first part of the dorsal
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    # The main part is connected with the residual part
    x = Concatenate()([x, route_1])

    # 1x1 convolution on chained results
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    feat = x
    x = Concatenate()([route, x])

    # Compression in height and width with maximum grouping
    x = MaxPooling2D(pool_size=[2,2],)(x)

    return x, feat

#---------------------------------------------------#
#   The main part of Cs lowercase pdarknet
#---------------------------------------------------#
def darknet_body(x):
    # First use two 3x3 convolutions with a 2x2 stride for height and width compression
    # 416,416.3 -> 208,208.32 -> 104,104,64
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2))(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2))(x)
    
    # 104,104,64 -> 52,52,128
    x, _ = resblock_body(x,num_filters = 64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x,num_filters = 128)
    # 26.26.256 -> x is 13.13.512
    #           -> Feat1 is 26.26.256
    x, feat1 = resblock_body(x,num_filters = 256)
    # 13,13,512 -> 13,13,512
    x = DarknetConv2D_BN_Leaky(512, (3,3))(x)

    feat2 = x
    return feat1, feat2

