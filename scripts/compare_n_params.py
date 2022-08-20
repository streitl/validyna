from ml_collections import config_flags
from validyna.registry import model_registry
from absl import app

_CONFIG = config_flags.DEFINE_config_file('cfg')


def main(_argv):
    cfg = _CONFIG.value

    for model_name, params in cfg.models:
        model = model_registry[model_name](
            n_in=cfg.n_in, n_features=cfg.n_features, space_dim=cfg.space_dim, n_out=cfg.n_out, **params
        )
        n_params_featurizer = sum(map(lambda p: p.numel(), model._get_featurizer_parameters()))
        n_params_total = sum(map(lambda p: p.numel(), model.parameters()))
        print(f'{model_name}: {n_params_featurizer} / {n_params_total}')


if __name__ == '__main__':
    app.run(main)
