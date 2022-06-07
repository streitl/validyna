import dysts.flows
from absl import app
from dysts.base import get_attractor_list
from ml_collections import config_flags

from validyna.data import generate_and_save_data_dictionary

_CONFIG = config_flags.DEFINE_config_file('cfg')


def main(_argv):
    cfg = _CONFIG.value
    attractors = cfg.attractors
    if cfg.attractors == 'all':
        attractors = get_attractor_list()

    attractors = [getattr(dysts.flows, a)() for a in attractors]
    if 'filter' in cfg:
        attractors = [a.name for a in attractors if cfg.filter(a)]

    common_string = '-'.join([f'{k}={v}' for k, v in sorted(cfg.data.common.items()) if k != 'verbose'])
    if 'individual' in cfg.data:
        for name, args in cfg.data.individual.items():
            individual_string = '-'.join([f'{k}={v}' for k, v in sorted(args.items())])
            save_location = f'{cfg.path}({common_string})/{name}({individual_string})'
            generate_and_save_data_dictionary(attractors, dir_path=save_location, **args, **cfg.data.common)
    else:
        save_location = f'{cfg.path}({common_string}))'
        generate_and_save_data_dictionary(attractors, dir_path=save_location, **cfg.data.common)


if __name__ == '__main__':
    app.run(main)
