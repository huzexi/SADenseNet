from os import path

from keras.layers import ReLU


class Config:
    # Directory paths
    dir_data = './data'
    dir_tmp = path.join('tmp')
    dir_tmp_test = path.join(dir_tmp, 'test')

    # Task properties
    a_in = (2, 2)
    a_out = (8, 8)
    a_syn = a_out[0] * a_out[1] - a_in[0] * a_in[1]

    # Network
    dense_a = True
    dense_s = True
    dense_i = True
    corr_block_n = 6
    corr_block_s = 5
    corr_block_a = 1
    corr_block_flt = 32

    @staticmethod
    def activation():
        return ReLU()
