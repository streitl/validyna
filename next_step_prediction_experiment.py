import os
import random

import dysts.flows
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from dysts.base import get_attractor_list
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from base.data import load_or_generate_and_save, build_in_out_pair_dataloader
from base.models import ForecasterLightning, LSTMForecaster, NBEATSForecaster

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if not os.path.isdir('results'):
    os.mkdir('results')

# The parameters of the experiment
trajectory_count = 1000
trajectory_length = 100  # 5000
data_params = dict(trajectory_count=trajectory_count, trajectory_length=trajectory_length)

n_in = 1
n_out = 1
common_model_params = dict(n_in=n_in, n_out=n_out)

n_epochs = 5
train_part = 0.75
n_splits = 2
batch_size = 64
num_workers = 8
experiment_params = dict(n_epochs=n_epochs, train_part=train_part, n_splits=n_splits, batch_size=batch_size,
                         num_workers=num_workers)

models_and_params = [
    (LSTMForecaster, {'n_hidden': 32, 'n_layers': 1}),
    (NBEATSForecaster, {'n_stacks': 4, 'n_blocks': 4, 'n_layers': 4, 'layer_widths': 5, 'expansion_coefficient_dim': 5})
]


class MetricLoggerCallback(pl.Callback):
    def __init__(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        def get_predictions(dataloader):
            return torch.cat([
                pl_module.model.forecast(batch[:, :model.n_in, :], n=batch.size(1) - pl_module.model.n_in)
                for (batch,) in dataloader
            ], dim=0)

        train_preds = get_predictions(self.train_dataloader)
        val_preds = get_predictions(self.val_dataloader)

        # TODO metrics on trajectories+
        trainer.logger.log_metrics({'lua': 0.1})


for attractor_idx, attractor_name in enumerate(get_attractor_list()):

    data_params['attractor'] = attractor_name
    attractor = getattr(dysts.flows, attractor_name)()

    attractor_x0 = attractor.ic.copy()
    space_dim = len(attractor_x0)

    data_params_string = '_'.join([f'{k}_{v}' for k, v in sorted(data_params.items(), key=lambda x: x[0])])

    print(f'Generating data for attractor {attractor_name}...')
    data = load_or_generate_and_save(
        f'data/{data_params_string}.pt',
        chaos_model=attractor,
        data_params=data_params,
        ic_fun=lambda: np.random.rand(space_dim) - 0.5 + attractor_x0
    )
    dataset = torch.utils.data.TensorDataset(data)
    print('Data generated/loaded')

    train_size = int(train_part * trajectory_count)
    val_size = trajectory_count - train_size

    for split in range(n_splits):
        print(f'Data split nº {split + 1} / {n_splits}')
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        chunk_train_dl = build_in_out_pair_dataloader(
            train_dataset,
            n_in=n_in,
            n_out=n_out,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )

        chunk_val_dl = build_in_out_pair_dataloader(
            val_dataset,
            n_in=n_in,
            n_out=n_out,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

        trajectory_train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        trajectory_val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        for Model, model_params in models_and_params:
            model = Model(n_features=space_dim, **model_params, **common_model_params)
            model_name = model.name()

            wandb_logger = pl.loggers.WandbLogger(
                save_dir='results',
                project='chaos-next-step-prediction',  # 'chaos-next-step-prediction',
                name=f'{model_name}_{attractor_name}_{split + 1}'
            )

            wandb_logger.experiment.config.update({
                'split_n': split + 1,
                'model.name': model_name,
                'model': model_params,
                'data': data_params,
                'experiment': experiment_params
            })

            model_trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=n_epochs,
                precision=32,
                callbacks=[MetricLoggerCallback(trajectory_train_dl, trajectory_val_dl)]
            )
            forecaster = ForecasterLightning(forecasting_model=model)
            model_trainer.fit(forecaster, train_dataloaders=chunk_train_dl, val_dataloaders=chunk_val_dl)

            wandb_logger.experiment.finish(quiet=True)
