from keras.layers import *

from components.layer import SpatialConv, AngularConv


def CorrelationBlock(flt, n_s, n_a, dense_s=False,
                     name='', activation=ReLU):
    def create(inp):
        x = inp
        dense_s_block = []
        for k in range(n_s):
            x = SpatialConv(flt, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation=activation,
                            name='%s_s%d' % (name, k)
                            )(x)
            if dense_s:
                dense_s_block.append(x)
                x = Concatenate(name='%s_dense_%d' % (name, k))(dense_s_block) \
                    if len(dense_s_block) > 1 else dense_s_block[0]
        for k in range(n_a):
            x = AngularConv(flt, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation=activation,
                            name='%s_a%d' % (name, k)
                            )(x)
        return x

    return create
