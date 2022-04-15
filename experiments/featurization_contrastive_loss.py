import os
import random
from itertools import product

import dysts.base
import dysts.flows
import numpy as np
import torch.utils.data
from numpy.random import rand
from torch.utils.data import TensorDataset, DataLoader

from config import ROOT_DIR
from ecodyna.data import generate_trajectories
from ecodyna.mutitask_models import MultiTaskRNN, MultiTaskNBEATS

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if not os.path.isdir(f'{ROOT_DIR}/results'):
        os.mkdir(f'{ROOT_DIR}/results')

    # data parameters
    dp = {'trajectory_count': 1000, 'trajectory_length': 100}
    # in out parameters (appear in many places)
    iop = {'n_in': 5}
    # common model parameters
    cmp = {**iop}
    # experiment parameters
    ep = {'n_epochs': 5, 'train_part': 0.75, 'n_splits': 2}
    # data loader parameters
    dlp = {'batch_size': 64, 'num_workers': 8, **iop}

    models_and_params = [
        (MultiTaskRNN, {'model': 'LSTM', 'n_hidden': 32, 'n_layers': 1}),
        (MultiTaskNBEATS,
         {'n_stacks': 4, 'n_blocks': 4, 'n_layers': 4, 'expansion_coefficient_dim': 5, 'layer_widths': 20})
    ]

    train_size = int(ep['train_part'] * dp['trajectory_count'])
    val_size = dp['trajectory_count'] - train_size

    datasets = {}
    for attractor_name in dysts.base.get_attractor_list():
        attractor = getattr(dysts.flows, attractor_name)()

        attractor_x0 = attractor.ic.copy()
        space_dim = len(attractor_x0)

        data = generate_trajectories(attractor, ic_fun=lambda: rand(space_dim) - 0.5 + attractor_x0, **dp)

        if space_dim not in datasets:
            datasets[space_dim] = []
        datasets[space_dim].append(TensorDataset(data))

    for space_dim in datasets.keys():

        for (i, dataset_a), (j, dataset_b) in product(enumerate(datasets[space_dim]), repeat=2):
            if i == j:
                continue

        # TODO
