from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'prober'
    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'task': 'forecasting',
            'trainer_callbacks': [('Prober', {'task': 'featurization'}, 'all')]
        },
        {
            'restore_run_suffix': 'forecasting',
            'task': 'featurization',
            'run_suffix': 'forecasting_then_featurization',
            'trainer_callbacks': [('Prober', {'task': 'forecasting'}, 'all')]
        },
        {
            'restore_run_suffix': 'forecasting',
            'task': 'classification',
            'run_suffix': 'forecasting_then_classification',
            'trainer_callbacks': [('Prober', {'task': 'forecasting'}, 'all')]
        }
    ]))
    return cfg
