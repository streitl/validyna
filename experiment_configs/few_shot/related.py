from experiment_configs.default import get_config as default_config
from experiment_configs.few_shot.common import get_config as common_config


def get_config():
    cfg = common_config()
    placeholders = default_config('placeholders')

    cfg.project = 'few-shot-sprott'

    placeholders['f_all'] = lambda a: a in {f'Sprott{letter}' for letter in ['A', 'B', 'C', 'D', 'E']}
    cfg.runs = cfg.runs[1:]
    cfg.runs[0]['run_suffix'] = 'related_lone'

    return cfg
