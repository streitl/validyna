from pytorch_lightning.callbacks import EarlyStopping
from scripts.experiments.defaults import all_models

from ecodyna.tasks.featurization import run_triplet_featurization_experiment

if __name__ == '__main__':
    params = {
        'experiment': {
            'project': 'lstm-forecasting-comparison',
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
            'common': {
                'n_features': 10,
                'n_layers': 1
            },
            'list': all_models
        },
        'dataloader': {
            'batch_size': 64,
            'num_workers': 8
        },
        'trainer': {
            'max_epochs': 50,
            'deterministic': True,
            'val_check_interval': 1 / 16,
            'callbacks': [EarlyStopping('val_loss', patience=3)]
        },
        'in_out': {
            'n_in': 5
        }
    }
    params['models']['common'].update(params['in_out'])

    run_triplet_featurization_experiment(params)
