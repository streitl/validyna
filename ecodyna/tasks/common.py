import os
from typing import Tuple

import pytorch_lightning as pl

from config import ROOT_DIR


def experiment_setup(params: dict) -> Tuple[int, int]:
    # Sets random seed for random, numpy and torch
    pl.seed_everything(params['experiment']['random_seed'], workers=True)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    train_size = int(params['experiment']['train_part'] * params['data']['trajectory_count'])
    val_size = params['data']['trajectory_count'] - train_size
    return train_size, val_size
