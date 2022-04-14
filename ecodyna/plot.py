from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def plot_3d_trajectories(trajectories: Sequence[Tensor], labels: Sequence[str], n_plots: int):
    if len(trajectories) != len(labels):
        raise ValueError(
            f'Number of trajectories and of labels must be equal (got {len(trajectories)} and {len(labels)})')

    all_tensors = torch.concat(*trajectories, dim=1)
    x_min, y_min, z_min = all_tensors.min(dim=(0, 1))
    x_max, y_max, z_max = all_tensors.max(dim=(0, 1))

    w = 4
    h = int(np.ceil(n_plots / w))

    fig = plt.figure(figsize=(h * 4, w * 4))

    ax = None
    for i in range(n_plots):
        ax = fig.add_subplot(h, w, i + 1, projection='3d')
        ic = trajectories[0][i, 0, :]
        ax.set_title(f'(x0, y0, z0)={tuple(np.round(ic, 2))}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        for n in range(len(trajectories)):
            ax.plot3D(*trajectories[n][i, :, :].T, label=labels[n])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    plt.tight_layout()
    return fig
