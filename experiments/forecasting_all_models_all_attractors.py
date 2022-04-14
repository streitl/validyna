import os
import random

import dysts.flows
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from dysts.base import get_attractor_list
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset

from config import ROOT_DIR
from ecodyna.data import load_or_generate_and_save, build_in_out_pair_dataloader
from ecodyna.metrics import ForecastMetricLogger
from ecodyna.mutitask_models import MultiTaskLSTM, MultiTaskNBEATS
from ecodyna.pl_wrappers import LightningForecaster

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    # data parameters
    dp = {'trajectory_count': 1000, 'trajectory_length': 100}
    # common model parameters
    cmp = {'n_in': 5, 'n_out': 5}
    # experiment parameters
    ep = {'n_epochs': 2, 'train_part': 0.75, 'n_splits': 2}
    # data loader parameters
    dlp = {'batch_size': 64, 'num_workers': 8}

    models_and_params = [
        (MultiTaskLSTM, {'n_hidden': 32, 'n_layers': 1}),
        (MultiTaskNBEATS,
         {'n_stacks': 4, 'n_blocks': 4, 'n_layers': 4, 'expansion_coefficient_dim': 5, 'layer_widths': 20})
    ]

    for attractor_idx, attractor_name in enumerate(get_attractor_list()):

        dp['attractor'] = attractor_name
        attractor = getattr(dysts.flows, attractor_name)()

        attractor_x0 = attractor.ic.copy()
        series_dim = len(attractor_x0)

        data_path = f"{ROOT_DIR}/data/{'_'.join([f'{k}_{v}' for k, v in sorted(dp.items(), key=lambda x: x[0])])}.pt"

        print(f'Generating data for attractor {attractor_name}...')
        data = load_or_generate_and_save(data_path, attractor=attractor, data_params=dp,
                                         ic_fun=lambda: np.random.rand(series_dim) - 0.5 + attractor_x0)
        dataset = TensorDataset(data)

        train_size = int(ep['train_part'] * dp['trajectory_count'])
        val_size = dp['trajectory_count'] - train_size

        for split in range(ep['n_splits']):
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            chunk_train_dl = build_in_out_pair_dataloader(train_dataset, **cmp, **dlp, shuffle=True)
            chunk_val_dl = build_in_out_pair_dataloader(val_dataset, **cmp, **dlp)

            for Model, model_params in models_and_params:
                model = Model(series_dim=series_dim, **model_params, **cmp)
                model_name = model.name()

                wandb_logger = WandbLogger(
                    save_dir=f'{ROOT_DIR}/results',
                    project='chaos-forecasting',
                    name=f'{model_name}_{attractor_name}_{split + 1}'
                )

                wandb_logger.experiment.config.update({
                    'split_n': split + 1,
                    'model.name': model_name,
                    'model': model_params,
                    'data': dp,
                    'dataloader': dlp,
                    'experiment': ep
                })

                model_trainer = pl.Trainer(logger=wandb_logger,
                                           max_epochs=ep['n_epochs'],
                                           callbacks=[ForecastMetricLogger(train_dataset, val_dataset)])
                forecaster = LightningForecaster(model=model)
                model_trainer.fit(forecaster, train_dataloaders=chunk_train_dl, val_dataloaders=chunk_val_dl)

                wandb_logger.experiment.finish(quiet=True)
