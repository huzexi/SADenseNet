import json
import numpy as np
import h5py
import random

from components.utils import sai_io_idx


class Generator:
    def __init__(self, dataset, config):
        self.config = config

        self.dataset = dataset

        self.batch_sz = config.train_batch_sz
        self.patch_sz = config.train_patch_sz
        self.iter_n = config.train_iter
        self.a_in = config.a_in
        self.a_out = config.a_out
        self.augment_config = config.train_aug

        self.h5 = h5py.File(dataset.get_path_train(), 'r',
                            'core' if config.train_use_memory else None)
        self.names = json.loads(self.h5.attrs['names'])

        # For training only
        self.batch_x = np.zeros([self.batch_sz,
                                 self.a_in[0], self.a_in[1],
                                 self.patch_sz[0], self.patch_sz[1]
                                 ])
        self.batch_y = np.zeros([self.batch_sz,
                                 self.a_out[0] * self.a_out[1] - self.a_in[0] * self.a_in[1],
                                 self.patch_sz[0], self.patch_sz[1]
                                 ])

        self.len = len(self.h5)

    def __len__(self):
        return self.iter_n

    def __getitem__(self, idx):
        for i in range(self.batch_sz):
            idx = random.randint(0, self.len - 1)
            name = self.names[idx]
            lf_crop = self.lf_crop(self.h5[name+'/ycrcb'], self.patch_sz)
            lf_crop = self.data_augment(lf_crop, self.augment_config)
            self.batch_x[i, :, :, :, :], self.batch_y[i, :, :, :] = self.get_xy(lf_crop, self.a_in)

        return self.batch_x, self.batch_y

    def __del__(self):
        self.h5.close()

    @classmethod
    def lf_crop(cls, h5, patch_sz):
        sz_s = h5.shape[2:4]
        sx = random.randint(0, sz_s[1] - patch_sz[1])
        sy = random.randint(0, sz_s[0] - patch_sz[0])

        lf_crop = h5[:, :, sy:sy + patch_sz[0], sx:sx + patch_sz[1], 0]  # Only need Y channel
        return lf_crop
    
    @classmethod
    def get_batch_xy(cls, lf, a_inp=(2, 2)):
        """ LF -> Batch x and y. For size>1 batch only, lf should have 5 dimensions. """
        batch_sz = lf.shape[0]
        a_sz = (lf.shape[1], lf.shape[2])
        s_sz = (lf.shape[3], lf.shape[4])
        in_sai, out_sai = sai_io_idx(lf.shape[1:3], a_inp)
        lf = lf.reshape([batch_sz, a_sz[0] * a_sz[1], s_sz[0], s_sz[1]])
        in_lf, out_lf = lf[:, in_sai, :, :], lf[:, out_sai, :, :]
        in_lf = in_lf.reshape([batch_sz, a_inp[0], a_inp[1], s_sz[0], s_sz[1]])
        return in_lf, out_lf

    @classmethod
    def get_xy(cls, lf, a_in):
        """ LF -> Batch x and y. For size=1 batch only, lf should have 4 dimensions. """
        a_sz = (lf.shape[0], lf.shape[1])
        s_sz = (lf.shape[2], lf.shape[3])
        in_sai, out_sai = sai_io_idx(lf.shape[0:2], a_in)
        lf = lf.reshape([a_sz[0] * a_sz[1], s_sz[0], s_sz[1]])
        in_lf, out_lf = lf[in_sai, :, :], lf[out_sai, :, :]
        in_lf = in_lf.reshape([a_in[0], a_in[1], s_sz[0], s_sz[1]])
        return in_lf, out_lf