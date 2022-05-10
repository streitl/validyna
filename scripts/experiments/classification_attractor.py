import os

import dysts.base
import dysts.flows
import pytorch_lightning as pl
import torch
from dysts.base import DynSysDelay
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm

from config import ROOT_DIR
from ecodyna.data import load_or_generate_and_save, build_slices
from ecodyna.mutitask_models import MyRNN, MyNBEATS, MyTransformer
from ecodyna.pl_wrappers import LightningClassifier

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
            'common': {},
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
            'callbacks': [EarlyStopping('val_loss', patience=5)]
        },
        'metric_loggers': [],
        'in_out': {
            'n_in': 5
        }
    }
    params['models']['common'].update(params['in_out'])

    # Sets random seed for random, numpy and torch
    pl.seed_everything(params['experiment']['random_seed'], workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    train_size = int(params['experiment']['train_part'] * params['data']['trajectory_count'])
    val_size = params['data']['trajectory_count'] - train_size

    attractors_per_dim = {}
    for attractor_name in dysts.base.get_attractor_list():
        attractor = getattr(dysts.flows, attractor_name)()

        # For speedup TODO remove
        if hasattr(attractor, '_postprocessing') or isinstance(attractor, DynSysDelay):
            continue

        space_dim = len(attractor.ic)

        if space_dim not in attractors_per_dim:
            attractors_per_dim[space_dim] = []
        attractors_per_dim[space_dim].append(attractor)

    for space_dim, attractors in list(attractors_per_dim.items()):
        datasets = {}
        print(f'Generating trajectories for attractors of dimension {space_dim}')
        for attractor in tqdm(attractors):
            attractor_x0 = attractor.ic.copy()
            datasets[attractor.name] = TensorDataset(load_or_generate_and_save(attractor, **params['data']))

        n_classes = len(attractors)

        for split in range(params['experiment']['n_splits']):

            train_datasets = []
            val_datasets = []
            for class_n, (attractor_name, dataset) in enumerate(datasets.items()):
                train_trajectories, val_trajectories = random_split(dataset, [train_size, val_size])
                X_train = build_slices(train_trajectories, **params['in_out'])
                X_val = build_slices(val_trajectories, **params['in_out'])

                y_train = torch.full(size=(len(X_train),), fill_value=class_n)
                y_val = torch.full(size=(len(X_val),), fill_value=class_n)

                train_datasets.append(TensorDataset(X_train, y_train))
                val_datasets.append(TensorDataset(X_val, y_val))

            train_dl = DataLoader(ConcatDataset(train_datasets), **params['dataloader'], shuffle=True)
            val_dl = DataLoader(ConcatDataset(val_datasets), **params['dataloader'])

            for Model, model_params in params['models']['list']:
                model = Model(space_dim=space_dim, n_classes=n_classes, **model_params, **params['models']['common'])

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project='classification-of-attractor',
                    name=f'{model.name()}_dim_{space_dim}_split_{split + 1}'
                )

                wandb_logger.experiment.config.update({
                    'split_n': split + 1,
                    'featurizer': {'name': model.name(), **model.hyperparams},
                    'data': params['data'],
                    'dataloader': params['dataloader'],
                    'experiment': params['experiment'],
                    'trainer': {k: f'{v}' for k, v in params['trainer'].items()}
                })

                model_trainer = pl.Trainer(logger=wandb_logger, **params['trainer'])
                classifier = LightningClassifier(model=model)
                model_trainer.fit(classifier, train_dataloaders=train_dl, val_dataloaders=val_dl)

                wandb_logger.experiment.finish(quiet=True)
