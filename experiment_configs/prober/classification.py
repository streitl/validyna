from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'prober'
    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'task': 'classification',
            'trainer_callbacks': [('Prober', {'task': 'featurization'}, 'all')]
        },
        {
            'restore_run_suffix': 'classification',
            'task': 'featurization',
            'run_suffix': 'classification_then_featurization',
            'trainer_callbacks': [('Prober', {'task': 'classification'}, 'all')]
        },
        {
            'restore_run_suffix': 'classification',
            'task': 'forecasting',
            'run_suffix': 'classification_then_forecasting',
            'trainer_callbacks': [('Prober', {'task': 'classification'}, 'all')]
        }
    ]))
    return cfg
