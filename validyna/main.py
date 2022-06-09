import os
from typing import Optional

import pytorch_lightning as pl
from absl import app
from ml_collections import ConfigDict, config_flags
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from validyna.data import ChunkMultiTaskDataset, load_data_dictionary
from validyna.models import multitask_models as mm
from validyna.registry import model_registry, task_registry

_CONFIG = config_flags.DEFINE_config_file('cfg')


def train_model_for_task(
        model: mm.MultiTaskTimeSeriesModel,
        task: str,
        datasets: dict[str, ChunkMultiTaskDataset],
        cfg: ConfigDict,
        run_suffix: Optional[str] = None
):
    if run_suffix is None:
        run_name = f'{model.name()}_{task}'
    else:
        run_name = f'{model.name()}_{run_suffix}'
    wandb = WandbLogger(project=cfg.project, name=run_name, id=run_name, save_dir=cfg.results_dir)
    wandb._id = f'{run_name}_{wandb.experiment.id}'
    module = task_registry[task](model=model, datasets=datasets, cfg=cfg)
    trainer = pl.Trainer(logger=wandb,
                         callbacks=[EarlyStopping(monitor=f'{module.loss_name}.val', **cfg.early_stopping),
                                    LearningRateMonitor(logging_interval='epoch')],
                         **cfg.trainer)
    trainer.fit(module)
    wandb.experiment.finish(quiet=True)


def run_experiment(cfg: ConfigDict):
    """
    Args:
        - project (str): the name of the project, passed to Weights and Biases (wandb)
        - task (str): the training task (name registered in {registry.model_registry})
        - pre_task (Optional[str]): a pre-training task (name registered in {registry.model_registry})
        - seed (int): the random seed to be used for the entire experiment
        - n_in (int): number of time steps that are given to the models
        - n_out (int): (if forecasting is involved) number of future time steps models predicted by the model
        - trainer (ConfigDict): the args are passed to <pl.Trainer>
        - data (ConfigDict):
            -
        - datasets (ConfigDict):
            - train: (dict[attractor_name, torch.Tensor])
            - val: (dict[attractor_name, torch.Tensor])
            - test: (dict[attractor_name, torch.Tensor])
        - models (dict[model_class_name, dict]): TODO
        - metrics (list): TODO
    """
    if 'pre_task' in cfg and cfg.pre_task == cfg.task:
        raise ValueError(f'The pre-training and training tasks are identical ({cfg.task})')

    if not os.path.isdir(cfg.results_dir):
        os.makedirs(os.path.dirname(cfg.results_dir), exist_ok=True)

    data = {name: load_data_dictionary(dir_path=path) for name, path in cfg.datasets.items()}
    datasets = {name: ChunkMultiTaskDataset(tensor_dict, cfg.n_in, cfg.n_out) for name, tensor_dict in data.items()}

    for model_name, model_args in cfg.models.items():
        pl.seed_everything(cfg.seed, workers=True)
        Model = model_registry[model_name]
        model = Model(n_in=cfg.n_in, n_features=cfg.n_features, space_dim=cfg.space_dim, **model_args)
        run_suffix = None
        if 'pre_task' in cfg:
            train_model_for_task(model, cfg.pre_task, datasets, cfg)
            model.freeze_featurizer()
            run_suffix = f'{cfg.pre_task}_{cfg.task}'
        train_model_for_task(model, cfg.task, datasets, cfg, run_suffix=run_suffix)


def main(_argv):
    run_experiment(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
