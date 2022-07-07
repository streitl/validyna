from experiment_configs.few_shot.common import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'few-shot-sprott'

    cfg.f_all = lambda a: a in [f'Sprott{letter}' for letter in ['A', 'B', 'C', 'D', 'E']]
    cfg.f_excluded = lambda a: a == 'SprottE'

    return cfg
