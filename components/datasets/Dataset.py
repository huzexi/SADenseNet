from abc import ABC, abstractmethod
from os import path


class Dataset(ABC):
    MODE_TRAIN = 1
    MODE_TEST = 2

    path_h5 = {
        MODE_TRAIN: '',
        MODE_TEST: ''
    }

    sz_a = (-1, -1)
    sz_s = (-1, -1)     # Spatial dimensions will be resized to this size when necessary, e.g. testing when training.

    def __init__(self, config):
        self.config = config

    def get_path(self, mode):
        return path.join(self.config.dir_data, self.path_h5[mode])

    def get_path_train(self):
        return self.get_path(self.MODE_TRAIN)

    def get_path_test(self):
        return self.get_path(self.MODE_TEST)

    @abstractmethod
    def prepare(self):
        """
        NOTICE: BGR should be uint8, YCrCb should float32.
        """
        raise NotImplementedError
