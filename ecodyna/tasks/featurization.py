from itertools import groupby

import dysts.base
import dysts.flows
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from config import ROOT_DIR
from ecodyna.data import TripletDataset, build_slices, load_from_params
from ecodyna.models.task_modules import ChunkTripletFeaturizer
from ecodyna.tasks.common import experiment_setup


def run_triplet_featurization_experiment(params: dict):
    train_size, val_size = experiment_setup(**params)

    attractors = [getattr(dysts.flows, attractor_name)() for attractor_name in params['experiment']['attractors']]
    attractors_per_dim = groupby(attractors, key=lambda x: len(x.ic))

    for space_dim, attractors in attractors_per_dim:
        print(f'Loading trajectories for attractors of dimension {space_dim}')
        datasets = {attractor.name: TensorDataset(load_from_params(attractor=attractor.name, **params['data']))
                    for attractor in tqdm(list(attractors))}

        for split_n in range(params['experiment']['n_splits']):
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

            triplet_train_dl = DataLoader(triplet_train_dataset, **params['dataloader'], shuffle=True)
            triplet_val_dl = DataLoader(triplet_val_dataset, **params['dataloader'])

            for Model, model_params in params['models']['all']:
                model = Model(space_dim=space_dim, **model_params, **params['models']['common'])

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project=params['experiment']['project'],
                    name=f'{model.name()}_dim_{space_dim}_split_{split_n}'
                )

                wandb_logger.experiment.config.update({
                    'split_n': split_n,
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