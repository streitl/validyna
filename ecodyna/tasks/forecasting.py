import os

import dysts.flows
import pytorch_lightning as pl
from dysts.base import DynSys
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import ROOT_DIR
from ecodyna.data import load_or_generate_and_save, build_in_out_pair_dataset
from ecodyna.models.task_modules import ChunkForecaster


def run_forecasting_experiment(params: dict):
    # Sets random seed for random, numpy and torch
    pl.seed_everything(params['experiment']['random_seed'], workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    train_size = int(params['experiment']['train_part'] * params['data']['trajectory_count'])
    val_size = params['data']['trajectory_count'] - train_size

    for attractor_name in params['experiment']['attractors']:
        attractor: DynSys = getattr(dysts.flows, attractor_name)()

        attractor_ic = attractor.ic.copy()
        space_dim = len(attractor_ic)

        data = load_or_generate_and_save(attractor=attractor, **params['data'])
        dataset = TensorDataset(data)

        for split in range(params['experiment']['n_splits']):
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            chunk_train_dl = DataLoader(build_in_out_pair_dataset(train_ds, **params['in_out']),
                                        **params['dataloader'], shuffle=True)
            chunk_val_dl = DataLoader(build_in_out_pair_dataset(val_ds, **params['in_out']),
                                      **params['dataloader'])

            # Add metric loggers to the list of trainer callbacks
            params['trainer']['callbacks'].extend([Logger(train_ds, val_ds) for Logger in params['metric_loggers']])

            for Model, model_params in params['models']['list']:
                model = Model(space_dim=space_dim, **model_params, **params['models']['common'])

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project=params['experiment']['project'],
                    name=f'{model.name()}_{attractor_name}_{split + 1}'
                )

                forecaster = ChunkForecaster(model=model)
                wandb_logger.experiment.config.update({
                    'split_n': split + 1,
                    'forecaster': {'name': model.name(), **forecaster.hparams},
                    'data': {'attractor': attractor_name, **params['data']},
                    'dataloader': params['dataloader'],
                    'experiment': params['experiment'],
                    'trainer': {k: f'{v}' for k, v in params['trainer'].items()}
                })

                trainer = pl.Trainer(logger=wandb_logger, **params['trainer'])
                trainer.fit(forecaster, train_dataloaders=chunk_train_dl, val_dataloaders=chunk_val_dl)

                wandb_logger.experiment.finish(quiet=True)
