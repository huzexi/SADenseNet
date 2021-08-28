from keras.layers import *

from components.layer.CorrelationBlock import CorrelationBlock


def DenseCorrelation(
    dense_s=True, dense_a=True, dense_i=True,
    corr_block_n=6, corr_block_s=5, corr_block_a=1,
    corr_block_flt=32,
    activation=ReLU):
    def create(inp):
        x = inp
        dense_a_block = []
        if dense_i:
            dense_a_block.append(inp)
        for i in range(corr_block_n):
            x = CorrelationBlock(flt=corr_block_flt, n_s=corr_block_s, n_a=corr_block_a,
                                 dense_s=dense_s,
                                 name='corr%d' % i, activation=activation)(x)
            if dense_a:
                dense_a_block.append(x)
            else:
                dense_a_block = [x]
                if dense_i:
                    dense_a_block.append(inp)
            x = Concatenate(name='dense%d_out' % i)(dense_a_block) if len(dense_a_block) > 1 else dense_a_block[0]

        return x

    return create
