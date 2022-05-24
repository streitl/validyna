import dysts.base

from ecodyna.tasks.classification import run_classification_of_attractors_experiment
from scripts.experiments.defaults import small_models

if __name__ == '__main__':
    params = {
        'experiment': {
            'attractors': dysts.base.get_attractor_list(),
            'project': 'classification-of-attractors',
            'train_part': 0.75,
            'random_seed': 42,
            'n_splits': 5
        },
        'data': {
            'trajectory_count': 100,
            'trajectory_length': 100,
            'resample': True,
            'pts_per_period': 50,
            'ic_noise': 0.01
        },
        'models': {
            'common': {'n_features': 32},
            'list': small_models
        },
        'dataloader': {
            'batch_size': 64,
            'num_workers': 8,
            'persistent_workers': True
        },
        'trainer': {
            'max_epochs': 100,
            'deterministic': True,
            'val_check_interval': 1 / 100,
            'limit_val_batches': 1 / 100,
            'log_every_n_steps': 50,
            'track_grad_norm': 2
        },
        'metric_loggers': [],
        'in_out': {
            'n_in': 5
        }
    }
    params['models']['common'].update(params['in_out'])
    params['data']['seed'] = params['experiment']['random_seed']

    run_classification_of_attractors_experiment(params)
