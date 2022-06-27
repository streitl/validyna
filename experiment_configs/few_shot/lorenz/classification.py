from experiment_configs.few_shot.lorenz.common import get_config as common_config


def get_config():
    cfg = common_config()
    cfg.tasks.common.task = 'classification'
    cfg.tasks.common.run = 'classification'
    return cfg
