from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'feature-freeze'
    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'task': 'forecasting',
        },
        {
            'task': 'classification',
            'freeze_featurizer': True,
            'run_suffix': 'forecasting_then_classification'
        }
    ]))
    return cfg
