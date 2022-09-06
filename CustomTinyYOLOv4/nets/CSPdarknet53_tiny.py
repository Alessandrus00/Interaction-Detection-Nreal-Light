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
#   Darknet convoluzione singola conv2 d
#   Se la dimensione del passaggio è 2, imposta tu stesso il metodo di riempimento.
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)
    
#---------------------------------------------------#
#   Blocco di convoluzione
#   DarknetConv2D + BatchNormalization + LeakyReLU
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
#   Blocchi strutturali di Cs pdarknet tiny
#   C'è un ampio margine residuo
#   Questo ampio bordo residuo bypassa molte strutture residue
#---------------------------------------------------#
def resblock_body(x, num_filters):
    # Utilizzare una convoluzione 3x3 per l'integrazione delle funzionalità
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    # portare a un ampio percorso marginale residuo
    route = x

    # I canali del Feature Layer vengono segmentati e la seconda parte viene considerata come la parte backbone.
    x = Lambda(route_group, arguments={'groups':2, 'group_id':1})(x) 
    # Convoluzione 3x3 sulla parte dorsale
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    # conduce ad un piccolo percorso marginale residuo 1
    route_1 = x
    # Convoluzione 3x3 sulla prima parte della dorsale
    x = DarknetConv2D_BN_Leaky(int(num_filters/2), (3,3))(x)
    # La parte principale è collegata con la parte residua
    x = Concatenate()([x, route_1])

    # Convoluzione 1x1 sui risultati concatenati
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    feat = x
    x = Concatenate()([route, x])

    # Compressione in altezza e larghezza con raggruppamento massimo
    x = MaxPooling2D(pool_size=[2,2],)(x)

    return x, feat

#---------------------------------------------------#
#   La parte principale di Cs pdarknet minuscola
#---------------------------------------------------#
def darknet_body(x):
    # Per prima cosa usa due convoluzioni 3x3 con falcata 2x2 per la compressione di altezza e larghezza
    # 416,416,3 -> 208,208,32 -> 104,104,64
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2))(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(64, (3,3), strides=(2,2))(x)
    
    # 104,104,64 -> 52,52,128
    x, _ = resblock_body(x,num_filters = 64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x,num_filters = 128)
    # 26.26.256 -> x è 13.13.512
    #           -> Feat1 è 26.26.256
    x, feat1 = resblock_body(x,num_filters = 256)
    # 13,13,512 -> 13,13,512
    x = DarknetConv2D_BN_Leaky(512, (3,3))(x)

    feat2 = x
    return feat1, feat2

