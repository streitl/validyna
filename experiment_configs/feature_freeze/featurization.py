from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.tasks.list = [
        {
            'task': 'featurization',
            'freeze_featurizer': True,
        },
        {
            'task': 'forecasting',
        },
        {
            'task': 'classification',
        }
    ]
    cfg.project = 'feature_freeze'
    return cfg
