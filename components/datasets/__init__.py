from .Dataset import Dataset
from .lytro.KalantariDataset import KalantariDataset
from .lytro.EpflDataset import EpflDataset
from .lytro.StanfordDataset import StanfordDataset


def get_dataset(name, config):
    # Lytro family
    if name == 'Kalantari' or name == '30Scenes':
        dataset = KalantariDataset(config)
    elif name == 'EPFL':
        dataset = EpflDataset(config)
    elif name == 'Stanford':
        dataset = StanfordDataset(config)
    elif name == 'reflective':
        dataset = StanfordDataset(config)
        dataset.set_category('reflective')
    elif name == 'occlusions':
        dataset = StanfordDataset(config)
        dataset.set_category('occlusions')
    else:
        raise ValueError("Unknown dataset.")

    return dataset
