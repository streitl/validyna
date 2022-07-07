from experiment_configs.default import get_config as default_config
from experiment_configs.few_shot.common import get_config as common_config


def get_config():
    cfg = common_config()
    placeholders = default_config('placeholders')

    cfg.project = 'few-shot-lorenz'

    placeholders['f_all'] = lambda a: 'Lorenz' in a
    placeholders['f_excluded'] = lambda a: a == 'Lorenz'

    return cfg
