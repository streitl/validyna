import dysts.flows
import pytorch_lightning as pl
from dysts.base import DynSys
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import ROOT_DIR
from ecodyna.data import build_in_out_pair_dataset, load_from_params
from ecodyna.models.task_modules import ChunkForecaster
from ecodyna.tasks.common import experiment_setup
from ecodyna.metrics import ForecastMetricLogger
from ecodyna.models.defaults import small_models


def forecasting(params: dict):
    train_size, val_size = experiment_setup(**params)

    for attractor_name in params['experiment']['attractors']:
        attractor = getattr(dysts.flows, attractor_name)()
        dataset = TensorDataset(load_from_params(attractor=attractor.name, **params['data']))
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        chunk_train_dl = DataLoader(build_in_out_pair_dataset(train_ds, **params['in_out']),
                                    **params['dataloader'], shuffle=True)
        chunk_val_dl = DataLoader(build_in_out_pair_dataset(val_ds, **params['in_out']),
                                  **params['dataloader'])

            # Add metric loggers to the list of trainer callbacks
            params['trainer']['callbacks'].extend([Logger(train_ds, val_ds) for Logger in params['metric_loggers']])

            for Model, model_params in params['models']['list']:
                model = Model(space_dim=len(attractor.ic), **model_params, **params['models']['common'])

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project=params['experiment']['project'],
                    name=f'{model.name()}_{attractor_name}'
                )

                forecaster = ChunkForecaster(model=model)
                wandb_logger.experiment.config.update({
                    'forecaster': {'name': model.name(), **forecaster.hparams},
                    'data': {'attractor': attractor_name, **params['data']},
                    'dataloader': params['dataloader'],
                    'experiment': params['experiment'],
                    'trainer': {k: f'{v}' for k, v in params['trainer'].items()}
                })

                trainer = pl.Trainer(logger=wandb_logger, **params['trainer'])
                trainer.fit(forecaster, train_dataloaders=chunk_train_dl, val_dataloaders=chunk_val_dl)

                wandb_logger.experiment.finish(quiet=True)


if __name__ == '__main__':
    params = {
        'experiment': {
            'attractors': dysts.base.get_attractor_list(),
            'project': 'forecasting',
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
            'list': small_models
        },
        'dataloader': {
            'batch_size': 64,
            'num_workers': 8
        },
        'trainer': {
            'max_epochs': 50,
            'deterministic': True,
            'val_check_interval': 1 / 16
        },
        'metric_loggers': [ForecastMetricLogger],
        'in_out': {
            'n_in': 5,
            'n_out': 5
        }
    }
    params['models']['common'].update(params['in_out'])

    forecasting(params=params)
