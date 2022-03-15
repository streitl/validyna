from typing import Callable
from darts import TimeSeries
from typing import Sequence

import numpy as np
from dysts.flows import DynSys

from utils import tensor_to_time_series, time_series_to_tensor


def generate_trajectories(
        chaos_model: DynSys,
        n_trajectories: int,
        trajectory_length: int,
        ic_fun: Callable[[], np.ndarray]
) -> Sequence[TimeSeries]:
    trajectories = []
    for n in range(n_trajectories):
        chaos_model.ic = ic_fun()
        trajectories.append(
            TimeSeries.from_values(
                chaos_model.make_trajectory(trajectory_length)
            )
        )
    return trajectories


def load_trajectories(path: str) -> Sequence[TimeSeries]:
    tensor: np.ndarray = np.load(path)
    return tensor_to_time_series(tensor)


def save_trajectories(trajectories: Sequence[TimeSeries], path: str):
    tensor: np.ndarray = time_series_to_tensor(trajectories)
    np.save(path, tensor)


def load_or_generate_and_save(
        path: str,
        chaos_model: DynSys,
        data_params: dict,
        ic_fun: Callable[[], np.ndarray]
) -> Sequence[TimeSeries]:
    try:
        trajectories = load_trajectories(path)
    except OSError:
        trajectories = generate_trajectories(
            chaos_model,
            n_trajectories=data_params['n_trajectories'],
            trajectory_length=data_params['trajectory_length'],
            ic_fun=ic_fun
        )
        save_trajectories(trajectories, path)

    return trajectories
