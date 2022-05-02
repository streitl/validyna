import os

import dysts.base
import dysts.flows
import pytorch_lightning as pl
from dysts.base import DynSysDelay
from numpy.random import rand
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from config import ROOT_DIR
from ecodyna.data import TripletDataset, build_sliced_dataset, load_or_generate_and_save, build_data_path
from ecodyna.mutitask_models import MultiTaskRNN
from ecodyna.pl_wrappers import LightningFeaturizer

if __name__ == '__main__':
    # Sets random seed for random, numpy and torch
    pl.seed_everything(42, workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    # data parameters
    dp = {'trajectory_count': 20, 'trajectory_length': 100, 'resample': True, 'pts_per_period': 50, 'ic_noise': 0.01}
    # in out parameters (appear in many places)
    iop = {'n_in': 10}
    # common model parameters
    cmp = {'n_features': 5, **iop}
    # experiment parameters
    ep = {'train_part': 0.75, 'n_splits': 2}
    # trainer parameters
    tp = {'max_epochs': 20}
    # data loader parameters
    dlp = {'batch_size': 64, 'num_workers': 8}

    models_and_params = [
        (MultiTaskRNN, {'model': 'GRU', 'n_layers': 1}),
        (MultiTaskRNN, {'model': 'LSTM', 'n_layers': 1})
    ]

    train_size = int(ep['train_part'] * dp['trajectory_count'])
    val_size = dp['trajectory_count'] - train_size

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

    for space_dim, attractors in attractors_per_dim.items():
        datasets = {}
        print(f'Generating trajectories for attractors of dimension {space_dim}')
        for attractor in tqdm(attractors):
            attractor_x0 = attractor.ic.copy()
            data = load_or_generate_and_save(build_data_path(**dp), attractor, **dp,
                                             ic_fun=lambda: dp['ic_noise'] * (rand(space_dim) - 0.5) + attractor_x0)
            datasets[attractor.name] = TensorDataset(data)

        for split in range(ep['n_splits']):
            print(f'Processing data into datasets')
            datasets = {
                attractor_name: dict(zip(['train', 'val'], random_split(dataset, [train_size, val_size])))
                for attractor_name, dataset in datasets.items()
            }
            train_datasets = {attractor_name: dataset['train'] for attractor_name, dataset in datasets.items()}
            val_datasets = {attractor_name: dataset['val'] for attractor_name, dataset in datasets.items()}

            chunk_train_datasets = {
                attractor_name: build_sliced_dataset(dataset, **iop)
                for attractor_name, dataset in train_datasets.items()
            }
            chunk_val_datasets = {
                attractor_name: build_sliced_dataset(dataset, **iop)
                for attractor_name, dataset in val_datasets.items()
            }
            triplet_train_dataset = TripletDataset(chunk_train_datasets)
            triplet_val_dataset = TripletDataset(chunk_val_datasets)

            triplet_train_dl = DataLoader(triplet_train_dataset, **dlp)
            triplet_val_dl = DataLoader(triplet_val_dataset, **dlp)

            for Model, mp in models_and_params:
                model = Model(space_dim=space_dim, **mp, **cmp)

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project='featurization-triplet-loss',
                    name=f'{model.name()}_dim_{space_dim}_split_{split + 1}'
                )

                wandb_logger.experiment.config.update({
                    'split_n': split + 1,
                    'featurizer': {'name': model.name(), **model.hyperparams},
                    'data': dp,
                    'dataloader': dlp,
                    'trainer': tp,
                    'experiment': ep
                })

                model_trainer = pl.Trainer(logger=wandb_logger, max_epochs=ep['max_epochs'], deterministic=True)
                forecaster = LightningFeaturizer(model=model)
                model_trainer.fit(forecaster, train_dataloaders=triplet_train_dl, val_dataloaders=triplet_val_dl)

                wandb_logger.experiment.finish(quiet=True)
