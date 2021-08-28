from components.datasets import get_dataset
from components.log import logger
from components.config import Config

if __name__ == '__main__':

    datasets = {
        # 'Kalantari',
        # 'Stanford',
        'EPFL',
    }
    for ds in datasets:
        logger.info("Start to prepare %s dataset." % ds)
        dataset = get_dataset(ds, Config)
        dataset.prepare()
