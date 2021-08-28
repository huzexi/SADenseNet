import keras
from keras.layers import *


def AngularConv(filters,
                kernel_size,
                strides=(1, 1),
                batch_norm=False,
                activation=ReLU,
                **kwargs):
    def create(inp):
        shape = keras.backend.int_shape(inp)
        sz_a = (shape[1], shape[2])
        sz_s = (shape[3], shape[4])
        x = Reshape((sz_a[0], sz_a[1], sz_s[0]*sz_s[1], -1))(inp)  # (b, a*a, s, s, c)

        # Angular Convolution
        x = Conv3D(filters, (kernel_size[0], kernel_size[1], 1),  # (b, a*a, s, s, c)
                   strides=(strides[0], strides[1], 1),
                   **kwargs)(x)

        x = activation()(x)
        if batch_norm:
            x = BatchNormalization()(x)

        shape = keras.backend.int_shape(x)
        sz_a = (shape[1], shape[2])
        x = Reshape((sz_a[0], sz_a[1], sz_s[0], sz_s[1], -1))(x)  # (b, a, a, s, s, c)
        return x

    return create
