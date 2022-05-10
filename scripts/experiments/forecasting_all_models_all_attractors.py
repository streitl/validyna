from pytorch_lightning.callbacks import EarlyStopping

from ecodyna.metrics import ForecastMetricLogger
from ecodyna.mutitask_models import MyRNN, MyNBEATS, MyTransformer
from scripts.experiments.forecasting_experiment import run_forecasting_experiment

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
            'common': dict(),
            'list': [
                (MyNBEATS, {
                    'n_stacks': 4, 'n_blocks': 2, 'expansion_coefficient_dim': 5, 'n_layers': 4, 'layer_widths': 16
                }),
                (MyTransformer, {}),
                (MyRNN, {'model': 'LSTM', 'n_hidden': 32, 'n_layers': 1}),
                (MyRNN, {'model': 'GRU', 'n_hidden': 32, 'n_layers': 1}),
            ]
        },
        'dataloader': {
            'batch_size': 64,
            'num_workers': 8
        },
        'trainer': {
            'max_epochs': 50,
            'deterministic': True,
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
