from typing import Sequence

import numpy as np
from darts import TimeSeries


def tensor_to_time_series(tensor: np.ndarray) -> Sequence[TimeSeries]:
    dims = tensor.shape
    if len(dims) == 2:
        return [TimeSeries.from_values(tensor)]
    elif len(dims) == 3:
        return [TimeSeries.from_values(tensor[n]) for n in range(dims[0])]
    else:
        raise ValueError(
            f'Given tensor does not correspond to one or to a sequence of time series'
            f' (found dim {len(dims)}, expected 2 or 3)'
        )


def time_series_to_tensor(series: Sequence[TimeSeries]) -> np.ndarray:
    return np.array([ts.all_values().squeeze() for ts in series])
