from ml_collections import ConfigDict

from config import ROOT_DIR


def get_config():
    cfg = ConfigDict()
    cfg.attractors = 'all'
    cfg.filter = lambda a: len(a.ic) == 3
    cfg.path = f'{ROOT_DIR}/data/default'
    cfg.data = ConfigDict()
    cfg.data.common = ConfigDict({
        'length': 200,
        'resample': True,
        'pts_per_period': 50,
        'seed': 2022,
        'verbose': True,
    })
    cfg.data.individual = ConfigDict({
        'train': {
            'count': 100,
            'ic_noise': 0.01,
            'ic_scale': 1
        },
        'val': {
            'count': 20,
            'ic_noise': 0.01,
            'ic_scale': 1
        },
        'test': {
            'count': 30,
            'ic_noise': 0.05,
            'ic_scale': 1.001
        }
    })
    return cfg
