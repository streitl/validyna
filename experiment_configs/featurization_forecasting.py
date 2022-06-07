from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.pre_task = 'featurization'
    cfg.task = 'forecasting'
    cfg.project = f'validyna-{cfg.pre_task}-to-{cfg.task}-cluster'
    return cfg