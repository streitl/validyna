import random
from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()

    cfg.sampling_ratio = 0.75

    all_datasets = cfg.datasets()
    cfg.runs = []
    for sampling_seed in range(5):
        random.seed(sampling_seed)
        datasets = {k: {a: t[random.sample(range(t.size(0)), int(cfg.sampling_ratio * t.size(0)))]
                        for a, t in v.items()}
                    for k, v in all_datasets.items()}
        cfg.runs.append(ConfigDict({
            'run_suffix': f'seed_{sampling_seed}',
            'datasets': lambda: datasets,
            'fresh_model': True,
        }))

    del cfg['datasets']
    return cfg
