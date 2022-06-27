import torch
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

from config import ROOT_DIR
from experiment_configs.default import get_config as default_config
from validyna.data import load_data_dictionary, ChunkMultiTaskDataset


def get_config():
    cfg = default_config()
    cfg.project = 'few-shot-lorenz'
    data_dir = f'{ROOT_DIR}/data/default(length=200-pts_per_period=50-resample=True-seed=2022)'
    data_dirs = {
        'train': f'{data_dir}/train(count=100-ic_noise=0.01-ic_scale=1)',
        'val': f'{data_dir}/val(count=20-ic_noise=0.01-ic_scale=1)',
        'test': f'{data_dir}/test(count=30-ic_noise=0.05-ic_scale=1.001)',
    }
    lorenz_datasets = {set_name: {attractor: data
                                  for attractor, data in load_data_dictionary(path).items()
                                  if 'Lorenz' in attractor}
                       for set_name, path in data_dirs.items()}

    cfg.tasks.list = [
        {
            'datasets': lambda: {
                set_name: ChunkMultiTaskDataset({attractor: (data if attractor != 'Lorenz' else torch.empty(0))
                                                 for attractor, data in data_dict.items()}, n_in=cfg.n_in,
                                                n_out=cfg.n_out)
                for set_name, data_dict in lorenz_datasets.items()
            },
        },
        {
            'datasets': lambda: {
                set_name: ChunkMultiTaskDataset(data_dict, n_in=cfg.n_in, n_out=cfg.n_out)
                for set_name, data_dict in lorenz_datasets.items()
            },
        }
    ]
    cfg.tasks.list = [ConfigDict(d) for d in cfg.tasks.list]
    return cfg
