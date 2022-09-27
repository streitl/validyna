from ml_collections import ConfigDict

from config import ROOT_DIR
from experiment_configs.default import get_config as default_config
from validyna.data import load_datasets


def get_config():
    cfg = default_config()

    data_dir = f'{ROOT_DIR}/data/default(length=250-pts_per_period=50-resample=True)'
    cfg.datasets = lambda: load_datasets({
        'train': f'{data_dir}/train(count=80-ic_noise=0.05-seed=0)',
        'val': f'{data_dir}/val(count=20-ic_noise=0.05-seed=1)',
        'test-52': f'{data_dir}/test-52(count=30-ic_noise=0.52-seed=20)',
        'test-56': f'{data_dir}/test-56(count=30-ic_noise=0.56-seed=30)',
        'test-63': f'{data_dir}/test-63(count=30-ic_noise=0.63-seed=31)',
        'test-72': f'{data_dir}/test-72(count=30-ic_noise=0.72-seed=75)',
        'test-85': f'{data_dir}/test-85(count=30-ic_noise=0.85-seed=92)',
        'test-100': f'{data_dir}/test(count=30-ic_noise=0.1-seed=2)',
    })
    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {}
    ]))

    return cfg
