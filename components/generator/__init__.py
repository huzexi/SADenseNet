from .Generator import Generator
from components.datasets import KalantariDataset


def get_generator(dataset, config):
    if type(dataset) is KalantariDataset:
        gen = Generator
    else:
        raise ValueError("Unknown generator.")

    return gen(dataset, config)
