import os
import random

import dysts.flows
import numpy as np
import torch
from dysts.base import get_attractor_list
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from darts.models import NBEATSModel, RNNModel, TransformerModel

from base.data import load_or_generate_and_save

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if not os.path.isdir('results'):
    os.mkdir('results')

data_params = {
    'trajectory_count': 10,  # 1000,
    'trajectory_length': 100  # 5000
}
common_model_params = {
    'input_chunk_length': 1,
    'output_chunk_length': 1
}
experiment_params = {
    'train_epochs': 5,
    'train_part': 0.75,
    'n_splits': 2
}

models_and_params = [
    ('GRU', RNNModel, {'model': 'GRU'}),
    ('LSTM', RNNModel, {'model': 'LSTM'}),
    ('NBEATS', NBEATSModel, {
        'num_blocks': 4,
        'num_stacks': 8,
        'layer_widths': 64
    }),
    ('Transformer', TransformerModel, {'dim_feedforward': 128}),
]

data_params_string = '_'.join([f'{k}_{v}' for k, v in data_params.items()])
common_params_string = '_'.join([f'{k}_{v}' for k, v in common_model_params.items()])
experiment_params_string = '_'.join([f'{k}_{v}' for k, v in experiment_params.items()])

for attractor_idx, attractor_name in enumerate(get_attractor_list()):

    data_params['attractor'] = attractor_name
    attractor = getattr(dysts.flows, attractor_name)()

    attractor_x0 = attractor.ic.copy()
    space_dim = len(attractor_x0)

    data_params_string = '_'.join([f'{k}_{v}' for k, v in sorted(data_params.items(), key=lambda x: x[0])])

    print(f'Generating data for attractor {attractor_name}...')
    data = load_or_generate_and_save(
        f'data/{data_params_string}.npy',
        chaos_model=attractor,
        data_params=data_params,
        ic_fun=lambda: np.random.rand(space_dim) - 0.5 + attractor_x0
    )
    print('Data generated/loaded')

    train_size = int(experiment_params['train_part'] * len(data))
    test_size = len(data) - train_size

    for split in range(experiment_params['n_splits']):
        print(f'Data split nº {split + 1} / {experiment_params["n_splits"]}')
        train_data, validation_data = torch.utils.data.random_split(data, [train_size, test_size])

        for model_name, Model, model_params in models_and_params:
            model = Model(**model_params, **common_model_params)

            wandb_logger = WandbLogger(
                save_dir='results',
                project='chaos-next-step-prediction-3',
                name=f'{model_name}_{attractor_name}_{split}'
            )

            wandb_logger.experiment.config.update(
                {
                    'id': f'{model_name}_{attractor_name}_{split}',
                    'split_n': split,
                    'model': model_name,
                    'model_params': model_params,
                    'data': data_params,
                    'experiment': experiment_params
                },
                allow_val_change=True
            )

            trainer = Trainer(
                logger=wandb_logger,
                precision=64,
                max_epochs=experiment_params['train_epochs'],
                enable_model_summary=True,
                enable_progress_bar=True,
                log_every_n_steps=5
            )
            model.fit(
                series=train_data,
                val_series=validation_data,
                epochs=experiment_params['train_epochs'],
                trainer=trainer
            )

            wandb_logger.experiment.finish(quiet=True)
