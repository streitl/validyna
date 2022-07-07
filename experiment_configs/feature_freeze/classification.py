from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'feature-freeze'
    cfg.tasks.list = [
        {
            'task': 'classification',
            'freeze_featurizer': True,
        },
        {
            'task': 'forecasting',
            'run': 'classification>forecasting'
        }
    ]
    cfg.tasks.list = [ConfigDict(d) for d in cfg.tasks.list]
    return cfg
