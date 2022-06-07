import os
import random
import re
import warnings
from typing import Callable, Optional, Union

import dysts.flows
import numpy as np
import torch
import tqdm
from dysts.flows import DynSys
from numpy.random import rand
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset


def generate_trajectories(
        attractor: DynSys,
        count: int,
        length: int,
        resample: bool,
        pts_per_period: int,
        verbose: bool = False,
        ic: Optional[Callable[[], np.ndarray]] = None,
        ic_noise: Optional[float] = None,
        ic_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs
) -> Tensor:
    attractor_ic = attractor.ic.copy()
    space_dim = len(attractor_ic)
    if ic is None:
        if count == 1:
            ic = lambda: attractor_ic
        elif ic_noise is not None:
            ic = lambda: ic_scale * attractor_ic + ic_noise * (rand(space_dim) - 0.5)
        else:
            raise ValueError('Without specifying an initial condition function, all trajectories will be identical')

    if seed is not None:
        np.random.seed(seed)

    trajectories = torch.empty(count, length, space_dim)
    if verbose:
        if len(kwargs) != 0:
            print(f'Warning: Ignoring arguments: {",".join(kwargs.keys())}')
        pbar = tqdm.tqdm(desc=attractor.name, leave=False, total=count, maxinterval=1)

    warnings.simplefilter('ignore', UserWarning)
    ics = []
    i = 0
    while i < count:
        attractor.ic = ic()
        tensor = Tensor(attractor.make_trajectory(n=length, resample=resample, pts_per_period=pts_per_period))
        # This might be false if the initial condition leads to a trajectory so extreme that
        # the integrator stops solving it after less than `length` steps
        if tensor.size(0) == length:
            trajectories[i, :, :] = tensor
            i += 1
            if verbose:
                pbar.update()
            ics.append(attractor.ic)
    pbar.close()
    warnings.simplefilter('default', UserWarning)
    return trajectories


def generate_data_dictionary(attractors: list[str], **kwargs) -> dict[str, Tensor]:
    return {name: generate_trajectories(getattr(dysts.flows, name)(), **kwargs) for name in attractors}


def generate_and_save_data_dictionary(attractors: list[str], dir_path: str, **kwargs) -> dict[str, Tensor]:
    data = dict()
    for name in attractors:
        path = f'{dir_path}/attractor={name}.pt'
        if os.path.isfile(path):
            continue
        data[name] = generate_trajectories(getattr(dysts.flows, name)(), **kwargs)
        save_trajectories(data[name], path=path)
    return data


def load_data_dictionary(dir_path: str) -> dict[str, Tensor]:
    result = dict()
    for filename in os.listdir(dir_path):
        attractor = re.search('attractor=(.+).pt', filename).group(0)
        result[attractor] = load_from_path(f'{dir_path}/{filename}')
    return result


def save_data_dictionary(data: dict[str, Tensor], dir_path: str):
    for name, tensor in data.items():
        save_trajectories(tensor, path=f'{dir_path}/attractor={name}.pt')


def load_from_path(path: str) -> Tensor:
    return torch.load(path)


def path_from_params(root_dir: str, **dp) -> str:
    return f"{root_dir}" \
           f"/data" \
           f"/{'-'.join([f'{k}={v}' for k, v in sorted(dp.items(), key=lambda x: x[0]) if k != 'attractor'])}" \
           f"/attractor={dp['attractor']}.pt"


def load_from_params(**dp) -> Tensor:
    path = path_from_params(**dp)
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
        self.trajectories_per_sys = trajectories_per_sys
        self.classes = {name: class_n for class_n, name in enumerate(sorted(trajectories_per_sys.keys()))}
        self.n_classes = len(self.classes)
        self.n_in = n_in
        self.n_out = n_out

        X_in = []
        X_out = []
        X_class = []
        for name, class_n in self.classes.items():
            trajectories = self.trajectories_per_sys[name]
            slices = build_slices(trajectories, n=n_in + n_out)
            X_in.append(slices[:, :n_in, :])
            X_out.append(slices[:, -n_out:, :])
            X_class.append(torch.full(size=(X_in[-1].size(0),), fill_value=class_n))

        self.X_in = torch.concat(X_in, dim=0)
        self.X_out = torch.concat(X_out, dim=0)
        self.X_class = torch.concat(X_class, dim=0)

        self.space_dim = self.X_in.size(2)

    def for_classification(self):
        return TensorDataset(self.X_in, self.X_class)

    def for_featurization(self):
        return TripletDataset(self.trajectories_per_sys)

    def for_forecasting(self):
        return TensorDataset(self.X_in, self.X_out)

    def for_all(self):
        return TensorDataset(self.X_in, self.X_class, self.X_out)


class TripletDataset(Dataset):

    def __init__(self, trajectories_per_sys: dict[str, Tensor]):
        assert len(trajectories_per_sys.keys()) > 1, 'There must be more than 1 class in the given dataset'
        self.tensors = trajectories_per_sys
        self.classes = list(sorted(self.tensors.keys()))
        self.class_sizes = {k: len(v) for k, v in self.tensors.items()}

    def __getitem__(self, index: int):
        # Anchor class is sampled uniformly TODO this could be a problem for unbalanced datasets
        anchor_class = self.classes[random.randint(0, len(self.classes) - 1)]
        # Negative class sampled randomly from other classes
        negative_class = [c for c in self.classes if c != anchor_class][random.randint(0, len(self.classes) - 2)]
        # Anchor and positive points are sampled randomly (could be the same)
        anchor_idx = random.randint(0, self.class_sizes[anchor_class] - 1)
        positive_idx = random.randint(0, self.class_sizes[anchor_class] - 1)
        negative_idx = random.randint(0, self.class_sizes[negative_class] - 1)
        anchor = self.tensors[anchor_class][anchor_idx]
        positive = self.tensors[anchor_class][positive_idx]
        negative = self.tensors[negative_class][negative_idx]
        return anchor, positive, negative

    def __len__(self):
        # TODO check if this makes any sense
        return sum(self.class_sizes.values())
