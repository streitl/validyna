from typing import Callable, Optional

import numpy as np
import torch
import tqdm
from dysts.flows import DynSys
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset


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


def build_in_out_pair_dataloader(
        dataset: Dataset,
        n_in: int,
        n_out: int,
        *args, **kwargs
) -> DataLoader:
    slices = torch.stack([
        tensor[i:i + n_in + n_out, :]
        for (tensor,) in dataset
        for i in range(tensor.size(0) - n_in - n_out)
    ])
    x = slices[:, :n_in, :]
    y = slices[:, n_in:, :]
    return DataLoader(TensorDataset(x, y), *args, **kwargs)
