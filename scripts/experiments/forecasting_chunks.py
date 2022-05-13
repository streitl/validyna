from pytorch_lightning.callbacks import EarlyStopping

from ecodyna.metrics import ForecastMetricLogger
from ecodyna.tasks.forecasting import run_forecasting_experiment
from scripts.experiments.defaults import all_models

if __name__ == '__main__':
    params = {
        'experiment': {
            'project': 'forecasting-performance',
            'train_part': 0.75,
            'random_seed': 26,
            'n_splits': 5
        },
        'data': {
            'trajectory_count': 100,
            'trajectory_length': 1000,
            'resample': True,
            'pts_per_period': 50,
            'ic_noise': 0.01
        },
        'models': {
            'common': {},
            'list': all_models
        },
        'dataloader': {
            'batch_size': 64,
            'num_workers': 8
        },
        'trainer': {
            'max_epochs': 50,
            'deterministic': True,
            'val_check_interval': 10,
            'callbacks': [EarlyStopping('val_loss', patience=3)]
        },
        'metric_loggers': [ForecastMetricLogger],
        'in_out': {
            'n_in': 5,
            'n_out': 5
        }
    }
    params['models']['common'].update(params['in_out'])

    run_forecasting_experiment(params=params)
