from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.task = 'featurization'
    cfg.project = f'validyna-{cfg.task}-cluster'
    return cfg
