from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from utils import time_series_to_tensor


def plot_3d_trajectories(trajectories: Sequence[Sequence[TimeSeries]], labels: Sequence[str], n_plots: int):

    if len(trajectories) != len(labels):
        raise ValueError('Number of trajectories and of labels must be equal')

    arrays = [time_series_to_tensor(ts) for ts in trajectories]

    xmin, ymin, zmin = np.concatenate(arrays, axis=1).min(axis=(0, 1))
    xmax, ymax, zmax = np.concatenate(arrays, axis=1).max(axis=(0, 1))

    w = 4
    h = int(np.ceil(n_plots / w))

    fig = plt.figure(figsize=(h * 4, w * 4))

    for i in range(n_plots):
        ax = fig.add_subplot(h, w, i + 1, projection='3d')
        ic = arrays[0][i, 0, :]
        ax.set_title(f'(x0, y0, z0)={tuple(np.round(ic, 2))}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        for n in range(len(trajectories)):
            ax.plot3D(*arrays[n][i, :, :].T, label=labels[n])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    plt.tight_layout()
    return fig


def plot_3d_predictions_vs_ground_truth(
        models: Sequence[TorchForecastingModel],
        ground_truth: Sequence[TimeSeries],
        data_params: dict,
        model_params: dict,
        n_plots: int
) -> [Sequence[Sequence[TimeSeries]]]:

    network_in = model_params['common']['input_chunk_length']
    trajectory_length = data_params['trajectory_length']

    start = [ts[pd.RangeIndex(0, network_in)] for ts in ground_truth]
    rest = [ts[pd.RangeIndex(network_in, trajectory_length)] for ts in ground_truth]
    predictions = []
    labels = []
    for model in models:
        predictions.append(model.predict(n=trajectory_length-network_in, series=start))
        labels.append(model.__class__.__name__)

    fig = plot_3d_trajectories([start, rest] + predictions, labels=['input', 'ground truth'] + labels, n_plots=n_plots)

    return predictions, fig
