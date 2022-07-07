import torch
from ml_collections import ConfigDict

from config import ROOT_DIR
from experiment_configs.default import get_config as default_config
from validyna.data import load_data_dictionary, ChunkMultiTaskDataset


def get_config():
    cfg = default_config()
    placeholders = default_config('placeholders')

    cfg.tasks.common.run = placeholders['tasks-common-task']
    del cfg.tasks.common.datasets

    data_dir = f'{ROOT_DIR}/data/default(length=200-pts_per_period=50-resample=True-seed=2022)'
    data_dirs = {
        'train': f'{data_dir}/train(count=100-ic_noise=0.01-ic_scale=1)',
        'val': f'{data_dir}/val(count=20-ic_noise=0.01-ic_scale=1)',
        'test': f'{data_dir}/test(count=30-ic_noise=0.05-ic_scale=1.001)',
    }

    cfg.f_all = lambda a: True
    cfg.f_excluded = lambda a: False

    datasets = {set_name: {attractor: data
                           for attractor, data in load_data_dictionary(path).items()
                           if cfg.f_all(attractor)}
                for set_name, path in data_dirs.items()}

    cfg.tasks.list = [
        {
            'datasets': lambda: {
                set_name: ChunkMultiTaskDataset({attractor: (torch.empty(0) if cfg.f_excluded(attractor) else data)
                                                 for attractor, data in data_dict.items()},
                                                n_in=cfg.n_in,
                                                n_out=cfg.n_out)
                for set_name, data_dict in datasets.items()
            },
            'run': 'before',
        },
        {
            'datasets': lambda: {
                set_name: ChunkMultiTaskDataset(data_dict, n_in=cfg.n_in, n_out=cfg.n_out)
                for set_name, data_dict in datasets.items()
            },
            'run': 'after',
        }
    ]
    cfg.tasks.list = [ConfigDict(d) for d in cfg.tasks.list]
    return cfg
