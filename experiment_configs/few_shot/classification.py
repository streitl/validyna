from ml_collections import ConfigDict

from config import ROOT_DIR
from experiment_configs.default import get_config as default_config
from validyna.data import load_data_dictionary


def get_config():
    cfg = default_config()
    data_dir = f'{ROOT_DIR}/data/default(length=200-pts_per_period=50-resample=True-seed=2022)'
    datasets = {
        k: load_data_dictionary(dir_path=path) for k, path in {
            'train': f'{data_dir}/train(count=100-ic_noise=0.01-ic_scale=1)',
            'val': f'{data_dir}/val(count=20-ic_noise=0.01-ic_scale=1)',
            'test': f'{data_dir}/test(count=30-ic_noise=0.05-ic_scale=1.001)',
        }.items()
    }

    cfg.tasks.list = [
        {
            'task': 'classification',
        },
        {
            'task': 'classification',
        }
    ]
    cfg.tasks.list = [ConfigDict(d) for d in cfg.tasks.list]
    cfg.project = 'few-shot'
    return cfg
