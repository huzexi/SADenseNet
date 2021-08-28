from keras import Model
from keras.backend import expand_dims
from keras.layers import *

from components.layer import AngularConv, SpatialConv
from components.layer.backend import DenseCorrelation


def create_model(patch_sz, config):
    inp = Input(shape=[*config.a_in, *patch_sz])
    x = Lambda(lambda x_: expand_dims(x_, -1))(inp)
    activation = config.activation

    x = DenseCorrelation(
        dense_s=config.dense_s,
        dense_a=config.dense_a,
        dense_i=config.dense_i,
        corr_block_n=config.corr_block_n,
        corr_block_s=config.corr_block_s,
        corr_block_a=config.corr_block_a,
        corr_block_flt=config.corr_block_flt,
        activation=activation
    )(x)
    x = SpatialConv(64, (3, 3),
                        strides=(1, 1),
                        padding='same',
                        activation=activation,
                        name='reduce0'
                        )(x)
    x = SpatialConv(64, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    activation=activation,
                    name='reduce1'
                    )(x)
    x = AngularConv(config.a_syn, (2, 2),
                    strides=(1, 1),
                    padding='valid',
                    activation=activation,
                    name='reduce2'
                    )(x)

    x = Reshape((patch_sz[0], patch_sz[1], config.a_syn))(x)
    out = Permute((3, 1, 2), name='out')(x)

    model = Model(inputs=inp, outputs=out)

    return model
