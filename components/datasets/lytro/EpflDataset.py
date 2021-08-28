import json
from os import path

import h5py

from components.log import logger
from components.datasets import Dataset
from components.datasets.lytro import load_item
from components.utils import path2img_name


class EpflDataset(Dataset):
    """
    Rerabek, M., Ebrahimi, T.: New light field image dataset.
    In: Proceedings of the 8th International Conference on Quality of Multimedia Experience.
    Number EPFL-CONF-218363(2016)
    """
    # Configuration
    path_raw = '/workplace/Datasets/LF/EPFL_PNG/'

    path_h5 = {
        Dataset.MODE_TEST: 'EPFL_test.h5'
    }
    list = {
        Dataset.MODE_TEST: ['Swans_2_eslf', 'ISO_Chart_7_eslf', 'Fountain_&_Vincent_1_eslf', 'Bench_in_Paris_eslf',
                            'Yan_&_Krios_standing_eslf', 'Friends_2_eslf', 'Diplodocus_eslf',
                            'Palais_du_Luxembourg_eslf', 'Flowers_eslf', 'Poppies_eslf', 'ISO_Chart_21_eslf',
                            'Sophie_&_Vincent_1_eslf', 'ISO_Chart_8_eslf', 'ISO_Chart_17_eslf',
                            'Sophie_&_Vincent_3_eslf', 'Overexposed_Stairs_eslf', 'Railway_Lines_2_eslf', 'Reeds_eslf',
                            'ISO_Chart_2_eslf', 'Perforated_Metal_1_eslf', 'Mirabelle_Prune_Tree_eslf', 'Bikes_eslf',
                            'Bridge_eslf', 'Magnets_1_eslf', 'Broken_Mirror_eslf', 'Overexposed_Sky_eslf',
                            'Caution_Bees_eslf', 'Fountain_&_Vincent_2_eslf', 'Fountain_2_eslf', 'ISO_Chart_3_eslf',
                            'Pond_in_Paris_eslf', 'Semi-reflecting_Structure_1_eslf', 'Yan_&_Krios_1_eslf',
                            'Spear_Fence_1_eslf', 'Ankylosaurus_&_Diplodocus_2_eslf', 'Sophie_Krios_&_Vincent_eslf',
                            'Wall_Decoration_eslf', 'Gravel_Garden_eslf', 'ISO_Chart_12_eslf', 'Pillars_eslf',
                            'ISO_Chart_15_eslf', 'Water_Drops_eslf', 'Vine_Wood_eslf', 'Wood_&_Net_eslf',
                            'Spear_Fence_2_eslf', 'Fountain_&_Bench_eslf', 'Tagged_Fence_eslf',
                            'Stone_Pillars_Outside_eslf', 'Desktop_eslf', 'Backlight_1_eslf', 'ISO_Chart_18_eslf',
                            'ISO_Chart_9_eslf', 'Ceiling_Light_eslf', 'Magnets_2_eslf', 'Trunk_eslf',
                            'Wheat_&_Silos_eslf', 'Backlight_2_eslf', 'ISO_Chart_14_eslf', 'Perforated_Metal_3_eslf',
                            'ISO_Chart_6_eslf', 'Danger_de_Mort_eslf', 'Graffiti_eslf', 'Rue_Gassendi_eslf',
                            'Sophie_&_Vincent_with_Flowers_eslf', 'Car_Dashboard_eslf', 'ISO_Chart_20_eslf',
                            'Sewer_Drain_eslf', 'Swans_1_eslf', 'Fountain_1_eslf', 'Semi-reflecting_Structure_2_eslf',
                            'Stone_Pillars_Inside_eslf', 'Rolex_Learning_Center_eslf', 'Friends_1_eslf',
                            'Friends_5_eslf', 'Sphynx_eslf', 'Railway_Lines_1_eslf', 'Parc_du_Luxembourg_eslf',
                            'Paved_Road_eslf', 'Geometric_Sculpture_eslf', 'Stairs_eslf', 'ISO_Chart_10_eslf',
                            'Books_eslf', 'ISO_Chart_5_eslf', 'Game_Board_eslf', 'ISO_Chart_23_eslf',
                            'Color_Chart_1_eslf', 'Black_Fence_eslf', 'Chain-link_Fence_2_eslf', 'Zwahlen_&_Mayr_eslf',
                            'ISO_Chart_13_eslf', 'ISO_Chart_16_eslf', 'Chain-link_Fence_1_eslf',
                            'Ankylosaurus_&_Diplodocus_1_eslf', 'ISO_Chart_11_eslf', 'Concrete_Cubes_eslf',
                            'Vespa_eslf', 'Bush_eslf', 'Perforated_Metal_2_eslf', 'Rusty_Handle_eslf',
                            'Houses_&_Lake_eslf', 'ISO_Chart_1_eslf', 'Ankylosaurus_&_Stegosaurus_eslf',
                            'ISO_Chart_19_eslf', 'University_eslf', 'ISO_Chart_4_eslf', 'Slab_&_Lake_eslf',
                            'Sophie_&_Vincent_on_a_Bench_eslf', 'Yan_&_Krios_2_eslf', 'Friends_3_eslf',
                            'Sophie_&_Vincent_2_eslf', 'Rusty_Fence_eslf', 'Color_Chart_2_eslf', 'Friends_4_eslf',
                            'ISO_Chart_22_eslf', 'Fountain_Pool_eslf', 'Billboards_eslf', 'Color_Chart_3_eslf',
                            'Red_&_White_Building_eslf']
    }

    sz_a_raw = (14, 14)
    sz_a = (8, 8)
    sz_s = (376, 540)

    def prepare(self):
        for mode, names in self.list.items():
            names.sort()
            h5_set = h5py.File(self.get_path(mode), 'w')
            names_short = [path2img_name(name) for name in names]
            h5_set.attrs['names'] = json.dumps(names_short)

            for idx, name in enumerate(names):
                logger.info("Preparing sample '%s'." % name)
                bgr, ycrcb = load_item(pth_img=path.join(self.path_raw, '%s.png' % name),
                                       a_raw=self.sz_a_raw, a_preserve=self.sz_a)

                h5_set.create_dataset(names_short[idx] + '/bgr', data=bgr)
                h5_set.create_dataset(names_short[idx] + '/ycrcb', data=ycrcb)

            h5_set.close()

