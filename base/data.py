from typing import Callable, Optional
from typing import Sequence

import numpy as np
import tqdm
from darts import TimeSeries
from dysts.flows import DynSys

from base.utils import tensor_to_time_series, time_series_to_tensor


def generate_trajectories(
        chaos_model: DynSys,
        trajectory_count: int,
        trajectory_length: int,
        ic_fun: Callable[[], np.ndarray]
) -> Sequence[TimeSeries]:
    trajectories = []
    for _ in tqdm.tqdm(range(trajectory_count)):
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
        ic_fun: Optional[Callable[[], np.ndarray]] = None
) -> Sequence[TimeSeries]:
    try:
        trajectories = load_trajectories(path)
    except OSError:
        trajectories = generate_trajectories(
            chaos_model,
            trajectory_count=data_params['trajectory_count'],
            trajectory_length=data_params['trajectory_length'],
            ic_fun=ic_fun
        )
        save_trajectories(trajectories, path)

    return trajectories
