import torch
from ml_collections import ConfigDict

from config import ROOT_DIR
from experiment_configs.default import get_config as default_config
from validyna.data import load_data_dictionary


def get_config():
    cfg = default_config()
    placeholders = default_config('placeholders')

    cfg.run_suffix = placeholders['task']
    del cfg['datasets']

    data_dir = f'{ROOT_DIR}/data/default(length=250-pts_per_period=50-resample=True)'
    data_dirs = {
        'train': f'{data_dir}/train(count=80-ic_noise=0.05-seed=0)',
        'val': f'{data_dir}/val(count=20-ic_noise=0.05-seed=1)',
        'test': f'{data_dir}/test(count=30-ic_noise=0.1-seed=2)',
    }

    placeholders['f_all'] = lambda a: True
    placeholders['f_excluded'] = lambda a: False

    def datasets():
        return {set_name: load_data_dictionary(path, placeholders['f_all']) for set_name, path in data_dirs.items()}

    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'datasets': lambda: {
                set_name: {attr: (torch.empty(0) if placeholders['f_excluded'](attr) else data)
                           for attr, data in data_dict.items()}
                for set_name, data_dict in datasets().items()
            },
            'run_suffix': 'before',
        },
        {
            'datasets': lambda: datasets(),
            'run_suffix': 'after',
            'trainer_callbacks': [('ClassSensitivitySpecificity', {'class_of_interest': 'SprottE'}, 'classification'),
                                  ('ClassFeatureSTD', {'class_of_interest': 'SprottE'}, 'featurization'),
                                  ('ClassMSE', {'class_of_interest': 'SprottE'}, 'forecasting')],
        }
    ]))
    return cfg
