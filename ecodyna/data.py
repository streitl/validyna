import random
from typing import Callable, Optional, Dict

import numpy as np
import torch
import tqdm
from dysts.flows import DynSys
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset


def generate_trajectories(
        attractor: DynSys,
        trajectory_count: int,
        trajectory_length: int,
        ic_fun: Optional[Callable[[], np.ndarray]] = None,
        verbose: bool = False
) -> Tensor:
    if ic_fun is None and trajectory_count > 1:
        raise ValueError('Without specifying an initial condition function, all trajectories will be identical')
    if verbose:
        print(f'Generating data for attractor {attractor.name}')
    trajectories = torch.empty(trajectory_count, trajectory_length, len(attractor.ic))

    for i in (tqdm.tqdm if verbose else lambda x: x)(range(trajectory_count)):
        attractor.ic = ic_fun()
        trajectories[i, :, :] = Tensor(attractor.make_trajectory(trajectory_length))
    return trajectories


def load_trajectories(path: str) -> Tensor:
    return torch.load(path)


def save_trajectories(trajectories: Tensor, path: str):
    torch.save(trajectories, path)


def load_or_generate_and_save(
        path: str,
        attractor: DynSys,
        trajectory_count: int,
        trajectory_length: int,
        ic_fun: Optional[Callable[[], np.ndarray]] = None
) -> Tensor:
    try:
        trajectories = load_trajectories(path)
    except OSError:
        trajectories = generate_trajectories(
            attractor,
            trajectory_count=trajectory_count,
            trajectory_length=trajectory_length,
            ic_fun=ic_fun
        )
        save_trajectories(trajectories, path)

    return trajectories


def build_sliced_dataset(dataset: Dataset, n_in: int) -> Dataset:
    slices = torch.stack([
        tensor[i:i + n_in, :]
        for (tensor,) in dataset
        for i in range(tensor.size(0) - n_in)
    ])
    return TensorDataset(slices)


def build_in_out_pair_dataset(dataset: Dataset, n_in: int, n_out: int) -> Dataset:
    slices = torch.stack([
        tensor[i:i + n_in + n_out, :]
        for (tensor,) in dataset
        for i in range(tensor.size(0) - n_in - n_out)
    ])
    x = slices[:, :n_in, :]
    y = slices[:, n_in:, :]
    return TensorDataset(x, y)


class TripletDataset(Dataset):

    def __init__(self, datasets_per_class: Dict[str, TensorDataset]):
        assert len(datasets_per_class.keys()) > 1, 'There must be more than 1 class in the given datasets'
        self.datasets = datasets_per_class
        self.classes = list(self.datasets.keys())
        self.class_sizes = {k: len(v) for k, v in self.datasets.items()}

    def __getitem__(self, index: int):
        # Anchor class is sampled deterministically and uniformly
        # TODO this could be a problem for unbalanced datasets
        anchor_class = self.classes[index % len(self.classes)]
        # Negative class sampled randomly from other classes
        negative_class = [c for c in self.classes if c != anchor_class][random.randint(0, len(self.classes) - 2)]
        # Anchor and positive points are sampled randomly (could be the same)
        anchor_idx = random.randint(0, self.class_sizes[anchor_class] - 1)
        positive_idx = random.randint(0, self.class_sizes[anchor_class] - 1)
        anchor = self.datasets[anchor_class][anchor_idx]
        positive = self.datasets[anchor_class][positive_idx]
        negative_idx = random.randint(0, self.class_sizes[negative_class] - 1)
        negative = self.datasets[negative_class][negative_idx]
        # print(f'{anchor_class} ({anchor_idx} and {positive_idx}) vs {negative_class} ({negative_idx})')
        # These are all tensor tuples of size 1
        return anchor[0], positive[0], negative[0]

    def __len__(self):
        # TODO check if this makes any sense
        # return sum([a * a * b for a in self.class_sizes.values() for b in self.class_sizes.values()])
        return sum(self.class_sizes.values())
