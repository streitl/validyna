from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'feature-freeze'
    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'task': 'forecasting',
            'freeze_featurizer': True,
        },
        {
            'task': 'classification',
            'run_suffix': 'forecasting>classification'
        }
    ]))
    return cfg
