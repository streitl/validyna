import random
import torch
from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'sampling'

    cfg.sampling_seed = 1
    cfg.sampling_ratio = 0.75

    random.seed(cfg.sampling_seed)

    data = cfg.datasets()

    cfg.datasets = lambda: {
        k: {a: t[random.sample(range(t.size(0)), int(cfg.sampling_ratio * t.size(0)))]
            for a, t in v.items()}
        for k, v in data.items()
    }
    cfg.runs = list(map(lambda d: ConfigDict(d), [{'run_suffix': f'seed_{cfg.sampling_seed}'}]))
    return cfg
