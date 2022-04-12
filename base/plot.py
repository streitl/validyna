from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_3d_trajectories(trajectories: Sequence[Sequence[np.ndarray]], labels: Sequence[str], n_plots: int):

    if len(trajectories) != len(labels):
        raise ValueError('Number of trajectories and of labels must be equal')

    arrays = trajectories
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
