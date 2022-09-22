from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()

    cfg.runs = list(map(lambda d: ConfigDict(d), [{'task': 'all'}]))
    cfg.loss_coefficients = {'mse': 0.8, 'cross': 0.1, 'triplet': 0.1}

    return cfg
