from math import sqrt
import numpy as np
import keras
from keras.backend import expand_dims
from keras.layers import *



def Reorder():
    """
    Chopping angular views into blocks for stride-4 conv (2x2). Example:
    From    [0, 1, 2,  3 ]  to  [0, 1, 4, 5]
            [4, 5, 6,  7 ]      [2, 3, 6, 7]
            [8, 9, 10, 11]      [8, 9, 12,13]
            [12,13,14, 15]      [10,11,14,15]
    """
    def create(inp):
        """
        :param inp: Size (batch_sz, a, a, s, s, 1)
        """
        x = inp
        shape = keras.backend.int_shape(x)
        sz_a = (shape[1], shape[2])
        sz_s = (shape[3], shape[4])

        x = Reshape((sz_a[0] * sz_a[1], sz_s[0], sz_s[1], -1))(x)
        shape = keras.backend.int_shape(x)
        sz_a = int(sqrt(shape[1]))
        inter_idx = []
        for i in range(0, sz_a, 4):  # Considering second conv, step is 4
            for j in range(0, sz_a, 4):
                idx = i * sz_a + j
                block = np.array([idx, idx + 1, idx + sz_a, idx + sz_a + 1] * 4)
                block[0:4] += 0
                block[4:8] += 2
                block[8:12] += 16
                block[12:16] += 18
                inter_idx += block.tolist()
        inter = []
        for i in inter_idx:
            inter.append(Lambda(lambda t: expand_dims(t[:, i, :, :], 1))(x))
        x = Concatenate(1)(inter)
        return x

    return create
