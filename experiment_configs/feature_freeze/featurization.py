from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'feature-freeze'
    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'task': 'featurization',
        },
        {
            'task': 'forecasting',
            'freeze_featurizer': True,
            'run_suffix': 'featurization_then_forecasting',
        },
        {
            'task': 'classification',
            'freeze_featurizer': True,
            'run_suffix': 'featurization_then_classification',
        }
    ]))
    return cfg
