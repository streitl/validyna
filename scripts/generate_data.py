import dysts.base
import dysts.flows
import pytorch_lightning as pl

from ecodyna.data import generate_trajectories, save_trajectories, build_data_path

if __name__ == '__main__':
    params = {
        'data': {
            'trajectory_count': 100,
            'trajectory_length': 100,
            'resample': True,
            'pts_per_period': 50,
            'ic_noise': 0.01,
            'seed': 42
        }
    }

    pl.seed_everything(params['data']['seed'])

    for attractor_name in dysts.base.get_attractor_list():
        trajectories = generate_trajectories(getattr(dysts.flows, attractor_name)(), **params['data'], verbose=True)
        save_trajectories(trajectories, path=build_data_path(attractor=attractor_name, **params['data']))
