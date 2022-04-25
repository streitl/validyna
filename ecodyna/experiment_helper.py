import os
from typing import Dict, Sequence, Tuple, Type, List

import dysts.flows
import pytorch_lightning as pl
from dysts.base import get_attractor_list, DynSys
from numpy.random import rand
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import ROOT_DIR
from ecodyna.data import load_or_generate_and_save, build_in_out_pair_dataset
from ecodyna.metrics import DatasetMetricLogger
from ecodyna.mutitask_models import MultiTaskTimeSeriesModel
from ecodyna.pl_wrappers import LightningForecaster


def run_forecasting_experiment(
        project: str,
        data_parameters: Dict,
        in_out_parameters: Dict,
        common_model_parameters: Dict,
        experiment_parameters: Dict,
        trainer_parameters: Dict,
        dataloader_parameters: Dict,
        models_and_params: Sequence[Tuple[Type[MultiTaskTimeSeriesModel], Dict]],
        metric_loggers: List[Type[DatasetMetricLogger]]
):
    dp = data_parameters
    iop = in_out_parameters
    cmp = common_model_parameters
    ep = experiment_parameters
    tp = trainer_parameters
    dlp = dataloader_parameters

    # Sets random seed for random, numpy and torch
    pl.seed_everything(42, workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    train_size = int(ep['train_part'] * dp['trajectory_count'])
    val_size = dp['trajectory_count'] - train_size

    for attractor_name in get_attractor_list():
        attractor: DynSys = getattr(dysts.flows, attractor_name)()

        attractor_x0 = attractor.ic.copy()
        space_dim = len(attractor_x0)

        data_path = f"{ROOT_DIR}/data/{'_'.join([f'{k}_{v}' for k, v in sorted(dp.items(), key=lambda x: x[0])])}.pt"

        data = load_or_generate_and_save(data_path, attractor, **dp,
                                         ic_fun=lambda: rand(space_dim) - 0.5 + attractor_x0)
        dataset = TensorDataset(data)

        for split in range(ep['n_splits']):
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            chunk_train_dl = DataLoader(build_in_out_pair_dataset(train_dataset, **iop), **dlp, shuffle=True)
            chunk_val_dl = DataLoader(build_in_out_pair_dataset(val_dataset, **iop), **dlp)

            tp['callbacks'] = (tp['callbacks'] if 'callbacks' in tp else []) + \
                [Logger(train_dataset, val_dataset) for Logger in metric_loggers]

            for Model, mp in models_and_params:
                model = Model(space_dim=space_dim, **mp, **cmp)

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project=project,
                    name=f'{model.name()}_{attractor_name}_{split + 1}'
                )

                wandb_logger.experiment.config.update({
                    'split_n': split + 1,
                    'forecaster': {'name': model.name(), **mp, **cmp},
                    'data': {'attractor': attractor_name, **dp},
                    'dataloader': dlp,
                    'experiment': ep
                })

                trainer = pl.Trainer(logger=wandb_logger, deterministic=True, **tp)
                forecaster = LightningForecaster(model=model)
                trainer.fit(forecaster, train_dataloaders=chunk_train_dl, val_dataloaders=chunk_val_dl)

                wandb_logger.experiment.finish(quiet=True)
