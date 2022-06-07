from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_3d_trajectories(tensors: List[Tensor], labels: List[str], n_plots: int):
    if len(tensors) != len(labels):
        raise ValueError(
            f'Number of trajectories and of labels must be equal (got {len(tensors)} and {len(labels)})')

    arrays = [tensor.numpy() for tensor in tensors]
    array_all = np.concatenate(tuple(arrays))
    x_min, y_min, z_min = np.min(array_all, axis=(0, 1))
    x_max, y_max, z_max = np.max(array_all, axis=(0, 1))

    w = 4
    h = int(np.ceil(n_plots / w))

    fig = plt.figure(figsize=(h * 4, w * 4))

    ax = None
    for i in range(n_plots):
        ax = fig.add_subplot(h, w, i + 1, projection='3d')
        ic = arrays[0][i, 0, :]
        ax.set_title(f'(x0,y0,z0)={tuple(np.round(ic, 1))}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        for n, array in enumerate(arrays):
            ax.plot3D(*array[i, :, :].T, label=labels[n])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    plt.tight_layout()
    return fig


def plot_1d_trajectories(tensors: List[Tensor], labels: List[str], n_plots: int):
    if len(tensors) != len(labels):
        raise ValueError(
            f'Number of trajectories and of labels must be equal (got {len(tensors)} and {len(labels)})')
    assert all(tensor.size(0) >= n_plots for tensor in tensors)

    arrays = [tensor.numpy() for tensor in tensors]
    array_all = np.concatenate(tuple(arrays))
    space_dim = array_all.shape[-1]
    mins = np.min(array_all, axis=(0, 1))
    maxs = np.max(array_all, axis=(0, 1))

    fig = plt.figure(figsize=(16, n_plots * space_dim))
    subfigs = fig.subfigures(n_plots, 1, hspace=0.01)

    for i in range(n_plots):
        ic = arrays[0][i, 0, :]
        axes = subfigs[i].subplots(space_dim, 1)
        subfigs[i].suptitle(f'x̅0={tuple(np.round(ic, 1))}')

        for n, array in enumerate(arrays):
            for dim, axis in enumerate(axes):
                axis.plot(array[i, :, dim], label=labels[n])
                axis.set_ylabel(f'x{dim}')
                axis.set_ylim(mins[dim], maxs[dim])
                axis.get_yaxis().set_label_coords(-0.03, 0.5)
                if dim + 1 != space_dim:
                    axis.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    plt.legend()
    return fig
