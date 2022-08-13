from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_3d_trajectories(tensors: List[Tensor], labels: List[str], n_plots: int, plot_titles: list[str], **kwargs):
    if len(tensors) != len(labels):
        raise ValueError(
            f'Number of trajectories and of labels must be equal (got {len(tensors)} and {len(labels)})')
    if len(plot_titles) != n_plots:
        raise ValueError(
            f'Number of plot titles and of plots must be equal (got {len(plot_titles)} and {n_plots})')

    arrays = [tensor.numpy() for tensor in tensors]
    array_all = np.concatenate(tuple(arrays))
    x_min, y_min, z_min = np.min(array_all, axis=(0, 1))
    x_max, y_max, z_max = np.max(array_all, axis=(0, 1))

    w = min(4, n_plots)
    h = int(np.ceil(n_plots / w))

    fig = plt.figure(figsize=(h * 4, w * 4))

    axes = []
    for i in range(n_plots):
        axes.append(fig.add_subplot(h, w, i + 1, projection='3d'))
        ax = axes[i]
        ax.set_title(plot_titles[i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        for n, array in enumerate(arrays):
            ax.plot3D(*array[i, :, :].T, label=labels[n], **kwargs)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    return fig, axes


def plot_1d_trajectories(tensors: List[Tensor], labels: List[str], n_plots: int, plot_titles: list[str], **kwargs):
    if len(tensors) != len(labels):
        raise ValueError(
            f'Number of trajectories and of labels must be equal (got {len(tensors)} and {len(labels)})')
    if len(plot_titles) != n_plots:
        raise ValueError(
            f'Number of plot titles and of plots must be equal (got {len(plot_titles)} and {n_plots})')
    assert all(tensor.size(0) >= n_plots for tensor in tensors)

    arrays = [tensor.numpy() for tensor in tensors]
    array_all = np.concatenate(tuple(arrays))
    space_dim = array_all.shape[-1]
    mins = np.min(array_all, axis=(0, 1))
    maxs = np.max(array_all, axis=(0, 1))

    fig = plt.figure(figsize=(16, n_plots * space_dim * 2))
    subfigs = fig.subfigures(n_plots, 1, hspace=0.01)

    axes = []
    for i in range(n_plots):
        ic = arrays[0][i, 0, :]
        subfig = subfigs[i] if n_plots > 1 else subfigs
        axes.append(subfig.subplots(space_dim, 1, sharex=True))
        subfig.suptitle(plot_titles[i])
        axes[i][-1].set_xlabel('time')

        for n, array in enumerate(arrays):
            for dim, ax in enumerate(axes[i]):
                ax.plot(array[i, :, dim], label=labels[n], **kwargs)
                ax.set_ylabel(f'x{dim}' if space_dim > 3 else ['x', 'y', 'z'][dim])
                ax.set_ylim(mins[dim], maxs[dim])
                ax.get_yaxis().set_label_coords(-0.03, 0.5)

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    return fig, axes
