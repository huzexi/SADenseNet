import json
from os import path

import h5py

from components.datasets.Dataset import Dataset
from components.datasets.lytro import load_item
from components.log import logger
from components.utils import path2img_name


class KalantariDataset(Dataset):
    """
    Kalantari, N.K., Wang, T.-C., Ramamoorthi, R., 2016.
    Learning-based view synthesis for light field cameras.
    ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2016) 35, 193.
    """

    # Configuration
    path_raw = '/workplace/Datasets/LF/Kalantari/'

    path_h5 = {
        # Dataset.MODE_TRAIN: 'Kalantari_train.h5',
        Dataset.MODE_TEST: 'Kalantari_test.h5'
    }
    list = {
        # Dataset.MODE_TRAIN: [
        #     'TrainingSet/STANFORD/bikes_11_eslf',
        #     'TrainingSet/STANFORD/bikes_12_eslf',
        #     'TrainingSet/STANFORD/bikes_13_eslf',
        #     'TrainingSet/STANFORD/bikes_20_eslf',
        #     'TrainingSet/STANFORD/bikes_4_eslf',
        #     'TrainingSet/STANFORD/bikes_9_eslf',
        #     'TrainingSet/STANFORD/buildings_10_eslf',
        #     'TrainingSet/STANFORD/buildings_3_eslf',
        #     'TrainingSet/STANFORD/buildings_6_eslf',
        #     'TrainingSet/STANFORD/cars_21_eslf',
        #     'TrainingSet/STANFORD/cars_36_eslf',
        #     'TrainingSet/STANFORD/cars_37_eslf',
        #     'TrainingSet/STANFORD/cars_38_eslf',
        #     'TrainingSet/STANFORD/cars_39_eslf',
        #     'TrainingSet/STANFORD/cars_44_eslf',
        #     'TrainingSet/STANFORD/cars_50_eslf',
        #     'TrainingSet/STANFORD/flowers_plants_17_eslf',
        #     'TrainingSet/STANFORD/flowers_plants_23_eslf',
        #     'TrainingSet/STANFORD/flowers_plants_24_eslf',
        #     'TrainingSet/STANFORD/flowers_plants_28_eslf',
        #     'TrainingSet/STANFORD/flowers_plants_42_eslf',
        #     'TrainingSet/STANFORD/flowers_plants_62_eslf',
        #     'TrainingSet/STANFORD/general_15_eslf',
        #     'TrainingSet/STANFORD/general_19_eslf',
        #     'TrainingSet/STANFORD/general_31_eslf',
        #     'TrainingSet/STANFORD/general_4_eslf',
        #     'TrainingSet/STANFORD/general_9_eslf',
        #     'TrainingSet/STANFORD/occlusions_24_eslf',
        #     'TrainingSet/OURS/IMG_0288_eslf',
        #     'TrainingSet/OURS/IMG_0289_eslf',
        #     'TrainingSet/OURS/IMG_0359_eslf',
        #     'TrainingSet/OURS/IMG_0360_eslf',
        #     'TrainingSet/OURS/IMG_0466_eslf',
        #     'TrainingSet/OURS/IMG_0518_eslf',
        #     'TrainingSet/OURS/IMG_0575_eslf',
        #     'TrainingSet/OURS/IMG_0596_eslf',
        #     'TrainingSet/OURS/IMG_0681_eslf',
        #     'TrainingSet/OURS/IMG_0780_eslf',
        #     'TrainingSet/OURS/IMG_0820_eslf',
        #     'TrainingSet/OURS/IMG_1016_eslf',
        #     'TrainingSet/OURS/IMG_1410_eslf',
        #     'TrainingSet/OURS/IMG_1413_eslf',
        #     'TrainingSet/OURS/IMG_1414_eslf',
        #     'TrainingSet/OURS/IMG_1415_eslf',
        #     'TrainingSet/OURS/IMG_1416_eslf',
        #     'TrainingSet/OURS/IMG_1419_eslf',
        #     'TrainingSet/OURS/IMG_1469_eslf',
        #     'TrainingSet/OURS/IMG_1470_eslf',
        #     'TrainingSet/OURS/IMG_1471_eslf',
        #     'TrainingSet/OURS/IMG_1473_eslf',
        #     'TrainingSet/OURS/IMG_1474_eslf',
        #     'TrainingSet/OURS/IMG_1475_eslf',
        #     'TrainingSet/OURS/IMG_1476_eslf',
        #     'TrainingSet/OURS/IMG_1477_eslf',
        #     'TrainingSet/OURS/IMG_1478_eslf',
        #     'TrainingSet/OURS/IMG_1479_eslf',
        #     'TrainingSet/OURS/IMG_1480_eslf',
        #     'TrainingSet/OURS/IMG_1481_eslf',
        #     'TrainingSet/OURS/IMG_1482_eslf',
        #     'TrainingSet/OURS/IMG_1483_eslf',
        #     'TrainingSet/OURS/IMG_1484_eslf',
        #     'TrainingSet/OURS/IMG_1486_eslf',
        #     'TrainingSet/OURS/IMG_1487_eslf',
        #     'TrainingSet/OURS/IMG_1490_eslf',
        #     'TrainingSet/OURS/IMG_1499_eslf',
        #     'TrainingSet/OURS/IMG_1500_eslf',
        #     'TrainingSet/OURS/IMG_1501_eslf',
        #     'TrainingSet/OURS/IMG_1504_eslf',
        #     'TrainingSet/OURS/IMG_1505_eslf',
        #     'TrainingSet/OURS/IMG_1508_eslf',
        #     'TrainingSet/OURS/IMG_1509_eslf',
        #     'TrainingSet/OURS/IMG_1510_eslf',
        #     'TrainingSet/OURS/IMG_1511_eslf',
        #     'TrainingSet/OURS/IMG_1513_eslf',
        #     'TrainingSet/OURS/IMG_1514_eslf',
        #     'TrainingSet/OURS/IMG_1516_eslf',
        #     'TrainingSet/OURS/IMG_1522_eslf',
        #     'TrainingSet/OURS/IMG_1523_eslf',
        #     'TrainingSet/OURS/IMG_1527_eslf',
        #     'TrainingSet/OURS/IMG_1529_eslf',
        #     'TrainingSet/OURS/IMG_1530_eslf',
        #     'TrainingSet/OURS/IMG_1534_eslf',
        #     'TrainingSet/OURS/IMG_1538_eslf',
        #     'TrainingSet/OURS/IMG_1544_eslf',
        #     'TrainingSet/OURS/IMG_1546_eslf',
        #     'TrainingSet/OURS/IMG_1547_eslf',
        #     'TrainingSet/OURS/IMG_1560_eslf',
        #     'TrainingSet/OURS/IMG_1565_eslf',
        #     'TrainingSet/OURS/IMG_1566_eslf',
        #     'TrainingSet/OURS/IMG_1567_eslf',
        #     'TrainingSet/OURS/IMG_1568_eslf',
        #     'TrainingSet/OURS/IMG_1580_eslf',
        #     'TrainingSet/OURS/IMG_1582_eslf',
        #     'TrainingSet/OURS/IMG_1583_eslf',
        #     'TrainingSet/OURS/IMG_1594_eslf',
        #     'TrainingSet/OURS/IMG_1595_eslf',
        #     'TrainingSet/OURS/IMG_1598_eslf',
        #     'TrainingSet/OURS/IMG_1599_eslf',
        #     'TrainingSet/OURS/IMG_1600_eslf',
        #     'TrainingSet/OURS/IMG_1601_eslf'
        # ],
        Dataset.MODE_TEST: [
            'TestSet/EXTRA/IMG_1085_eslf',
            'TestSet/EXTRA/IMG_1086_eslf',
            'TestSet/EXTRA/IMG_1184_eslf',
            'TestSet/EXTRA/IMG_1187_eslf',
            'TestSet/EXTRA/IMG_1306_eslf',
            'TestSet/EXTRA/IMG_1312_eslf',
            'TestSet/EXTRA/IMG_1316_eslf',
            'TestSet/EXTRA/IMG_1317_eslf',
            'TestSet/EXTRA/IMG_1320_eslf',
            'TestSet/EXTRA/IMG_1321_eslf',
            'TestSet/EXTRA/IMG_1324_eslf',
            'TestSet/EXTRA/IMG_1325_eslf',
            'TestSet/EXTRA/IMG_1327_eslf',
            'TestSet/EXTRA/IMG_1328_eslf',
            'TestSet/EXTRA/IMG_1340_eslf',
            'TestSet/EXTRA/IMG_1389_eslf',
            'TestSet/EXTRA/IMG_1390_eslf',
            'TestSet/EXTRA/IMG_1411_eslf',
            'TestSet/EXTRA/IMG_1419_eslf',
            'TestSet/EXTRA/IMG_1528_eslf',
            'TestSet/EXTRA/IMG_1541_eslf',
            'TestSet/EXTRA/IMG_1554_eslf',
            'TestSet/EXTRA/IMG_1555_eslf',
            'TestSet/EXTRA/IMG_1586_eslf',
            'TestSet/EXTRA/IMG_1743_eslf',
            'TestSet/PAPER/Cars',
            'TestSet/PAPER/Flower1',
            'TestSet/PAPER/Flower2',
            'TestSet/PAPER/Rock',
            'TestSet/PAPER/Seahorse'
        ]
    }

    sz_a_raw = (14, 14)
    sz_a = (8, 8)
    sz_s = (376, 540)

    def prepare(self):
        for mode, names in self.list.items():
            h5_set = h5py.File(self.get_path(mode), 'w')  # ds = dataset
            names_short = [path2img_name(name) for name in names]
            h5_set.attrs['names'] = json.dumps(names_short)

            for idx, name in enumerate(names):
                logger.info("Preparing sample '%s'." % name)
                bgr, ycrcb = load_item(pth_img=path.join(self.path_raw, '%s.png' % name),
                                       a_raw=self.sz_a_raw, a_preserve=self.sz_a)

                h5_set.create_dataset(names_short[idx] + '/bgr', data=bgr)
                h5_set.create_dataset(names_short[idx] + '/ycrcb', data=ycrcb)

            h5_set.close()

