from experiment_configs.default import get_config as default_config


def get_config():
    cfg = default_config()
    cfg.project = 'test_new_engine_classification'
    cfg.task = 'classification'
    return cfg
