import dysts.flows
import pytorch_lightning as pl
from dysts.base import DynSys
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import ROOT_DIR
from ecodyna.data import build_in_out_pair_dataset, load_from_params
from ecodyna.models.task_modules import ChunkForecaster
from ecodyna.tasks.common import experiment_setup


def run_forecasting_experiment(params: dict):
    train_size, val_size = experiment_setup(params)

    for attractor_name in params['experiment']['attractors']:
        attractor: DynSys = getattr(dysts.flows, attractor_name)()
        dataset = TensorDataset(load_from_params(attractor=attractor.name, **params['data']))

        for split_n in range(params['experiment']['n_splits']):
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
                    name=f'{model.name()}_{attractor_name}_{split_n}'
                )

                forecaster = ChunkForecaster(model=model)
                wandb_logger.experiment.config.update({
                    'split_n': split_n,
                    'forecaster': {'name': model.name(), **forecaster.hparams},
                    'data': {'attractor': attractor_name, **params['data']},
                    'dataloader': params['dataloader'],
                    'experiment': params['experiment'],
                    'trainer': {k: f'{v}' for k, v in params['trainer'].items()}
                })

                trainer = pl.Trainer(logger=wandb_logger, **params['trainer'])
                trainer.fit(forecaster, train_dataloaders=chunk_train_dl, val_dataloaders=chunk_val_dl)

                wandb_logger.experiment.finish(quiet=True)
