import os

import dysts.base
import dysts.flows
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from config import ROOT_DIR
from ecodyna.data import TripletDataset, build_slices, load_or_generate_and_save
from ecodyna.models.task_modules import ChunkTripletFeaturizer


def run_triplet_featurization_experiment(params: dict):
    # Sets random seed for random, numpy and torch
    pl.seed_everything(params['experiment']['random_seed'], workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    train_size = int(params['experiment']['train_part'] * params['data']['trajectory_count'])
    val_size = params['data']['trajectory_count'] - train_size

    attractors_per_dim = {}
    for attractor_name in params['data']['attractors']:
        attractor = getattr(dysts.flows, attractor_name)()

        space_dim = len(attractor.ic)

        if space_dim not in attractors_per_dim:
            attractors_per_dim[space_dim] = []
        attractors_per_dim[space_dim].append(attractor)

    for space_dim, attractors in attractors_per_dim.items():
        datasets = {}
        print(f'Generating trajectories for attractors of dimension {space_dim}')
        for attractor in tqdm(attractors):
            attractor_x0 = attractor.ic.copy()
            data = load_or_generate_and_save(attractor, **params['data'])
            datasets[attractor.name] = TensorDataset(data)

        for split in range(params['experiment']['n_splits']):
            datasets = {
                attractor_name: dict(zip(['train', 'val'], random_split(dataset, [train_size, val_size])))
                for attractor_name, dataset in datasets.items()
            }
            train_datasets = {attractor_name: dataset['train'] for attractor_name, dataset in datasets.items()}
            val_datasets = {attractor_name: dataset['val'] for attractor_name, dataset in datasets.items()}

            chunk_train_datasets = {
                attractor_name: build_slices(dataset, **params['in_out'])
                for attractor_name, dataset in train_datasets.items()
            }
            chunk_val_datasets = {
                attractor_name: build_slices(dataset, **params['in_out'])
                for attractor_name, dataset in val_datasets.items()
            }
            triplet_train_dataset = TripletDataset(chunk_train_datasets)
            triplet_val_dataset = TripletDataset(chunk_val_datasets)

            triplet_train_dl = DataLoader(triplet_train_dataset, **params['dataloader'])
            triplet_val_dl = DataLoader(triplet_val_dataset, **params['dataloader'])

            for Model, model_params in params['models']['all']:
                model = Model(space_dim=space_dim, **model_params, **params['models']['common'])

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project=params['experiment']['project'],
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

                model_trainer = pl.Trainer(logger=wandb_logger, deterministic=True, **params['trainer'])
                featurizer = ChunkTripletFeaturizer(model=model)
                model_trainer.fit(featurizer, train_dataloaders=triplet_train_dl, val_dataloaders=triplet_val_dl)

                wandb_logger.experiment.finish(quiet=True)