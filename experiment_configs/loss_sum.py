from ml_collections import ConfigDict

from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()

    cfg.runs = list(map(lambda d: ConfigDict(d), [
        {
            'task': 'all',
            'run_suffix': f'{p:.2f}',
            'loss_coefficients': {'mse': p, 'cross': (1-p) / 2, 'triplet': (1-p) / 2},
            'fresh_model': True,
        }
        for p in [1 / 3, 0.5, 2 / 3, 0.8]
    ]))

    cfg.models = list(filter(lambda t: t[0] == 'GRU', cfg.models))

    return cfg
