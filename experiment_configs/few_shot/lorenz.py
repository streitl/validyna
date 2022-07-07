from experiment_configs.few_shot.common import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'few-shot-lorenz'

    cfg.f_all = lambda a: 'Lorenz' in a
    cfg.f_excluded = lambda a: a == 'Lorenz'

    return cfg
