import os
import re
import warnings
from copy import deepcopy
from typing import Callable, Optional, Union, Tuple

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
    """
    Generates trajectories from the given dynamical system using random sampling of initial conditions.
    """
    sample = attractor.make_trajectory(n=500, resample=True, pts_per_period=100)
    mins = sample.min(axis=0)
    maxs = sample.max(axis=0)
    ic_noise = ic_noise * (maxs - mins)

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
    if verbose:
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


def load_data_dictionary(dir_path: str, cond: Callable[[str], bool] = lambda s: True) -> dict[str, Tensor]:
    print(f'Loading {dir_path}')
    result = dict()
    for filename in os.listdir(dir_path):
        attractor = re.search('attractor=(.+).pt', filename).group(1)
        if cond(attractor):
            result[attractor] = load_from_path(os.path.join(dir_path, filename))
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


def load_datasets(paths: dict[str, str]):
    return {name: load_data_dictionary(dir_path=path) for name, path in paths.items()}


def normalize(data: Tensor, mean: Tensor = torch.zeros(1), std: Tensor = torch.ones(1)) -> Tensor:
    # To avoid dividing by zero
    std[std == 0] = 1
    return (data - mean) / std


def scale_trajectory_group(trajectories: Tensor, mins: Optional[Tensor] = None, maxs: Optional[Tensor] = None):
    """
    Scales the trajectories per dimension component to be in the range [-1, 1].
    Args:
        - trajectories: an N x T x D tensor containing N trajectories of length T in a space of dimension D
        - mins: a D-dimensional tensor of minimums
        - max: a D-dimensional tensor of maximums
    """
    if mins is None and maxs is None:
        mins = trajectories.amin(dim=(0, 1))
        maxs = trajectories.amax(dim=(0, 1))
    D = trajectories.size(-1)
    m = mins.size(0)
    M = maxs.size(0)
    assert D == m and m == M, 'Trajectories and mins / maxs must have same dimension D'
    return (trajectories - mins) / (maxs - mins) * 2 - 1


class SliceMultiTaskDataset:
    """
    A dataset object containing information relevant to classification, feature extraction and forecasting tasks, namely
    the input time series, class information, and output time series.

    Here, trajectories of length T are split into slices of length n_in to be fed to the models.

    Args:
        - trajectories_per_class: a dictionary where keys are classes (i.e. attractor names) and the values are torch
            | Tensors of dimension N, T, D where N is the number of trajectories, T is the length of each trajectory,
            | and D is the vector size for each time step (must be the same for all classes!)
        - n_in (int): number of time steps that are given to the models
        - n_out (int): (if forecasting is involved) number of future time steps predicted by the model
    """

    def __init__(self, trajectories_per_class: dict[str, Tensor], n_in: int, n_out: int):
        empty_classes = [k for k, v in trajectories_per_class.items() if v.size(0) == 0]
        self.classes = [c for c in sorted(trajectories_per_class.keys()) if c not in empty_classes] + \
            list(sorted(empty_classes))
        self.n_classes = len(self.classes)
        self.n_non_empty_classes = self.n_classes - len(empty_classes)

        self.n_in = n_in
        self.n_out = n_out

        sample = next(iter(trajectories_per_class.values()))
        self.n_trajectories, self.trajectory_length, self.space_dim = tuple(sample.size())

        self.trajectories_per_class = deepcopy(trajectories_per_class)
        self.slices_per_class = None
        self.n_slices_per_class = self.n_trajectories * (self.trajectory_length - (self.n_in + self.n_out))

        self.X_in = None
        self.X_out = None
        self.X_class = None

        self.data_processed = False

    def get_mins_maxs(self) -> Tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Returns a dict with the minimum values in each dimension for each class, and another dict with the maximums
        """
        mins = dict()
        maxs = dict()
        for class_n, class_name in enumerate(self.classes):
            trajectories = self.trajectories_per_class[class_name]
            mins[class_name] = trajectories.amin(dim=(0, 1))
            maxs[class_name] = trajectories.amax(dim=(0, 1))
        return mins, maxs

    def process_data(self, mins: Optional[dict[str, Tensor]] = None, maxs: Optional[dict[str, Tensor]] = None):
        """
        Splits each trajectory into slices, creates class labels from dictionary, and creates
        """
        if self.data_processed:
            return
        X_in, X_out, X_class = [], [], []
        self.slices_per_class = dict()
        for class_n, class_name in enumerate(self.classes):
            trajectories = self.trajectories_per_class[class_name]
            if trajectories.size(0) == 0:
                continue
            if mins is not None and maxs is not None:
                trajectories = scale_trajectory_group(trajectories, mins=mins[class_name], maxs=maxs[class_name])
                self.trajectories_per_class[class_name] = trajectories
            slices = build_slices(trajectories, n=self.n_in + self.n_out)
            self.slices_per_class[class_name] = slices
            X_in.append(slices[:, :self.n_in, :])
            X_out.append(slices[:, -self.n_out:, :])
            X_class.append(torch.full(size=(slices.size(0),), fill_value=class_n))

        self.X_in = torch.concat(X_in, dim=0)
        self.X_out = torch.concat(X_out, dim=0)
        self.X_class = torch.concat(X_class, dim=0)

        self.data_processed = True

    def get_positive_negative_batch(self, anchor_classes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        For each anchor class in the batch, sample a negative class, then sample a trajectory slice for the anchor and
        negative classes.
        """
        if not self.data_processed:
            raise ValueError('Process data before calling get_positive_negative_batch')
        # Get batch of random negative classes different to the anchor ones
        other_classes = torch.randint_like(input=anchor_classes, high=self.n_non_empty_classes - 1)
        other_classes[other_classes >= anchor_classes] += 1
        nspc = self.n_slices_per_class
        positive = self.X_in[anchor_classes * nspc + torch.randint_like(anchor_classes, nspc)]
        negative = self.X_in[other_classes * nspc + torch.randint_like(anchor_classes, nspc)]
        return positive, negative

    def tensor_dataset(self) -> Dataset:
        if not self.data_processed:
            raise ValueError('Process data before calling tensor_dataset')
        return TensorDataset(self.X_in, self.X_out, self.X_class)
