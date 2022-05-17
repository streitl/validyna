import os

import dysts.base
import dysts.flows
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm

from config import ROOT_DIR
from ecodyna.data import load_or_generate_and_save, build_slices
from ecodyna.models.task_modules import ChunkClassifier


def run_classification_of_attractors_experiment(params: dict):
    # Sets random seed for random, numpy and torch
    pl.seed_everything(params['experiment']['random_seed'], workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    train_size = int(params['experiment']['train_part'] * params['data']['trajectory_count'])
    val_size = params['data']['trajectory_count'] - train_size

    attractors_per_dim = {}
    for attractor_name in dysts.base.get_attractor_list():
        attractor = getattr(dysts.flows, attractor_name)()

        space_dim = len(attractor.ic)

        if space_dim not in attractors_per_dim:
            attractors_per_dim[space_dim] = []
        attractors_per_dim[space_dim].append(attractor)

    for space_dim, attractors in list(attractors_per_dim.items()):
        datasets = {}
        print(f'Generating trajectories for attractors of dimension {space_dim}')
        for attractor in tqdm(attractors):
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
            val_dl = DataLoader(ConcatDataset(val_datasets), **params['dataloader'], shuffle=True)

            for Model, model_params in params['models']['list']:
                model = Model(space_dim=space_dim, n_classes=n_classes, **model_params, **params['models']['common'])
                classifier = ChunkClassifier(model=model)

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project=params['experiment']['project'],
                    name=f'{model.name()}_dim_{space_dim}_split_{split + 1}'
                )

                wandb_logger.experiment.config.update({
                    'split_n': split + 1,
                    'classifier': {'name': model.name(), **model.hyperparams},
                    'data': params['data'],
                    'dataloader': params['dataloader'],
                    'experiment': params['experiment'],
                    'trainer': {k: f'{v}' for k, v in params['trainer'].items()}
                })

                model_trainer = pl.Trainer(logger=wandb_logger, **params['trainer'])
                model_trainer.fit(classifier, train_dataloaders=train_dl, val_dataloaders=val_dl)

                wandb_logger.experiment.finish(quiet=True)
