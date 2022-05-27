import os
import random
from typing import Callable, Optional, Tuple, Union, Literal

import numpy as np
import torch
import tqdm
from dysts.flows import DynSys
from numpy.random import rand
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset, random_split, ConcatDataset

from config import ROOT_DIR


def generate_trajectories(
        attractor: DynSys,
        trajectory_count: int,
        trajectory_length: int,
        ic_fun: Optional[Callable[[], np.ndarray]] = None,
        verbose: bool = False,
        resample: bool = True,
        pts_per_period: int = 100,
        ic_noise: Optional[float] = None,
        **kwargs
) -> Tensor:
    if ic_fun is None and trajectory_count > 1:
        if ic_noise is None:
            raise ValueError('Without specifying an initial condition function, all trajectories will be identical')
        else:
            attractor_ic = attractor.ic.copy()
            ic_fun = lambda: ic_noise * (rand(len(attractor_ic)) - 0.5) + attractor_ic
    if verbose:
        print(f'Generating data for attractor {attractor.name}')
    trajectories = torch.empty(trajectory_count, trajectory_length, len(attractor.ic))

    for i in (tqdm.tqdm if verbose else lambda x: x)(range(trajectory_count)):
        attractor.ic = ic_fun()
        trajectories[i, :, :] = Tensor(
            attractor.make_trajectory(trajectory_length, resample=resample, pts_per_period=pts_per_period)
        )
    return trajectories


def build_data_path(**dp) -> str:
    return f"{ROOT_DIR}" \
           f"/data" \
           f"/{'-'.join([f'{k}={v}' for k, v in sorted(dp.items(), key=lambda x: x[0]) if k != 'attractor'])}" \
           f"/attractor={dp['attractor']}.pt"


def load_from_path(path: str) -> Tensor:
    return torch.load(path)


def load_from_params(**dp) -> Tensor:
    path = build_data_path(**dp)
    return load_from_path(path)


def save_trajectories(trajectories: Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(trajectories, path)


# Used for featurization and classification
def build_slices(data: Union[Tensor, TensorDataset], n: int) -> Tensor:
    if isinstance(data, TensorDataset):
        data = data.tensors[0]
    slices = torch.stack([
        tensor[i:i + n, :]
        for tensor in data
        for i in range(tensor.size(0) - n)
    ])
    return slices


# Used for forecasting
def build_in_out_pair_dataset(dataset: TensorDataset, n_in: int, n_out: int) -> Dataset:
    slices = build_slices(dataset, n=n_in + n_out)
    X = slices[:, :n_in, :]
    y = slices[:, n_in:, :]
    return TensorDataset(X, y)


class ChunkMultiTaskDataset:

    def __init__(self, trajectories_per_sys: dict[str, Tensor], n_in: int, n_out: int):
        self.classes = {name: class_n for class_n, name in enumerate(trajectories_per_sys.keys())}
        self.n_classes = len(self.classes)
        self.n_in = n_in
        self.n_out = n_out

        X_in = []
        X_out = []
        X_class = []
        for class_n, trajectories in enumerate(trajectories_per_sys.values()):
            slices = build_slices(trajectories, n=n_in + n_out)
            X_in.append(slices[:, :n_in, :])
            X_out.append(slices[:, -n_out:, :])
            X_class.append(torch.full(size=(X_in[-1].size(0),), fill_value=class_n))

        self.X_in = torch.concat(X_in, dim=0)
        self.X_out = torch.concat(X_out, dim=0)
        self.X_class = torch.concat(X_class, dim=0)

    def for_classification(self):
        return TensorDataset(self.X_in, self.X_class)

    def for_featurization(self):
        return TensorDataset(self.X_in)

    def for_forecasting(self):
        return TensorDataset(self.X_in, self.X_out)

    def for_all(self):
        return TensorDataset(self.X_in, self.X_class, self.X_out)


class ChunkClassDataset:

    def __init__(self, datasets: dict[str, Dataset], train_size: int, val_size: int, n_in: int):
        self.datasets = datasets
        self.classes = {name: class_n for class_n, name in enumerate(self.datasets.keys())}
        self.n_classes = len(self.classes)
        self.train_size = train_size
        self.val_size = val_size
        self.n_in = n_in

    def random_split(self) -> Tuple[Dataset, Dataset]:
        train_datasets = []
        val_datasets = []
        for class_n, dataset in enumerate(self.datasets.values()):
            train_trajectories, val_trajectories = random_split(dataset, [self.train_size, self.val_size])
            X_train = build_slices(train_trajectories, n=self.n_in)
            X_val = build_slices(val_trajectories, n=self.n_in)

            y_train = torch.full(size=(len(X_train),), fill_value=class_n)
            y_val = torch.full(size=(len(X_val),), fill_value=class_n)

            train_datasets.append(TensorDataset(X_train, y_train))
            val_datasets.append(TensorDataset(X_val, y_val))

        return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


class TripletDataset(Dataset):

    def __init__(self, tensors_per_class: dict[str, Tensor]):
        assert len(tensors_per_class.keys()) > 1, 'There must be more than 1 class in the given datasets'
        self.tensors = tensors_per_class
        self.classes = list(self.datasets.keys())
        self.class_sizes = {k: len(v) for k, v in self.tensors.items()}

    def __getitem__(self, index: int):
        # Anchor class is sampled deterministically and uniformly
        # TODO this could be a problem for unbalanced datasets
        anchor_class = self.classes[index % len(self.classes)]
        # Negative class sampled randomly from other classes
        negative_class = [c for c in self.classes if c != anchor_class][random.randint(0, len(self.classes) - 2)]
        # Anchor and positive points are sampled randomly (could be the same)
        anchor_idx = random.randint(0, self.class_sizes[anchor_class] - 1)
        positive_idx = random.randint(0, self.class_sizes[anchor_class] - 1)
        anchor = self.tensors[anchor_class][anchor_idx]
        positive = self.tensors[anchor_class][positive_idx]
        negative_idx = random.randint(0, self.class_sizes[negative_class] - 1)
        negative = self.tensors[negative_class][negative_idx]
        return anchor, positive, negative

    def __len__(self):
        # TODO check if this makes any sense
        # return sum([a * a * b for a in self.class_sizes.values() for b in self.class_sizes.values()])
        return sum(self.class_sizes.values())
