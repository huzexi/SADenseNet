import json
from itertools import chain
from os import path
import h5py

from components.log import logger
from components.datasets.Dataset import Dataset
from components.datasets.lytro import load_item


# noinspection PyTypeChecker
class StanfordDataset(Dataset):
    """
    Raj, A.S., Lowney, M., Shah, R., Wetzstein, G.: Stanford lytro light field archive.
    http://lightfields.stanford.edu/LF2016.html
    """

    # Configuration
    path_raw = '/workplace/Datasets/LF/Stanford/'

    path_h5 = {
        Dataset.MODE_TEST: 'Stanford_%s_test.h5'
    }
    list = {
        Dataset.MODE_TEST: {
            'occlusions': chain(range(1, 16), range(17, 18), range(20, 24), range(25, 33), range(35, 44), range(45, 51)),
            'reflective': chain(range(1, 15), range(16, 33)),
        }
    }

    sz_a_raw = (14, 14)
    sz_a = (8, 8)
    sz_s = (376, 540)

    def __init__(self, config):
        super().__init__(config)
        self.category = None

    def set_category(self, category):
        self.category = category

    def prepare(self):
        for mode, categories in self.list.items():
            for cate, rng in categories.items():
                names = [('%s_%d_eslf' % (cate, i)) for i in rng]
                h5_cate = h5py.File(self.get_path(mode, cate), 'w')
                h5_cate.attrs['names'] = json.dumps(names)

                for idx, name in enumerate(names):
                    logger.info("Preparing sample '%s'." % name)
                    bgr, ycrcb = load_item(pth_img=path.join(self.path_raw, cate, 'raw', '%s.png' % name),
                                           a_raw=self.sz_a_raw,
                                           a_preserve=self.sz_a)

                    h5_cate.create_dataset(name + '/bgr', data=bgr)
                    h5_cate.create_dataset(name + '/ycrcb', data=ycrcb)

                h5_cate.close()

    def get_path(self, mode, category=None):
        if category:
            return path.join(self.config.dir_data, self.path_h5[mode] % category)
        else:
            return path.join(self.config.dir_data, self.path_h5[mode] % self.category)
