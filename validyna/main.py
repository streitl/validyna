import os
from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
from absl import app
from ml_collections import ConfigDict, config_flags
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from validyna.data import ChunkMultiTaskDataset, make_datasets
from validyna.models import multitask_models as mm
from validyna.registry import task_registry

_CONFIG = config_flags.DEFINE_config_file('cfg')


def train_model_for_task(
        model: mm.MultiTaskTimeSeriesModel,
        task: str,
        datasets: dict[str, ChunkMultiTaskDataset],
        cfg: ConfigDict,
        run_suffix: Optional[str] = None
):
    """
    TODO
    Args:
        - model (str):
        - task (str):
        - datasets (dict[str, ChunkMultiTaskDataset):
        - cfg (ConfigDict): as specified in `run_experiment`, using only the keys
            - seed
            - use_wandb
            - project
            - results_dir
            - trainer
            - early_stopping
        - run_suffix (str)
    """
    pl.seed_everything(cfg.seed, workers=True)
    trainer_kwargs = {k: v for k, v in cfg.trainer.items() if k != 'callbacks'}
    if cfg.get('use_wandb', False):
        run_name = f'{model.name()}_{run_suffix if run_suffix is not None else task}'
        wandb_logger = WandbLogger(project=cfg.project, name=run_name, id=run_name, save_dir=cfg.results_dir)
        trainer_kwargs['logger'] = wandb_logger
    module = task_registry[task](model=model, datasets=datasets, cfg=cfg)
    trainer_callbacks = deepcopy(cfg.trainer.get('callbacks', []))
    trainer_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if 'early_stopping' in cfg:
        trainer_callbacks += [EarlyStopping(monitor=f'{module.loss_name}.val', **cfg.early_stopping)]
    trainer = pl.Trainer(callbacks=trainer_callbacks, **trainer_kwargs)
    trainer.fit(module)
    if cfg.get('use_wandb', False):
        trainer_kwargs['logger'].experiment.finish(quiet=True)


def run_experiment(cfg: ConfigDict):
    """
    Args:
        - cfg (ConfigDict): a configuration dictionary with the following keys:
            - results_dir (str): the path of the directory to save the results
            - seed (int): the random seed to be used for the entire experiment for each model training
            - use_wandb (bool): whether to log results using the Weights and Biases logger
            - project (Optional[str]): the name of the project, passed to wandb if used
            - n_in (int): number of time steps that are given to the models
            - n_out (int): (if forecasting is involved) number of future time steps models predicted by the model
            - trainer (ConfigDict): the items are passed to <pl.Trainer>
            - models (list[Tuple[str, dict]]): tuples where the first is a model name registered in the model registry,
             and the second is a dictionary with the parameters to be passed to the model
            - tasks (ConfigDict):
                - common (ConfigDict):
                    - datasets (Optional[ConfigDict]): has at least the following keys:
                        - train (str): the path to data used to optimize the model
                        - val (str): the path to data used for early stopping and learning rate adjustment
                    other keys will be treated as test sets and metrics will be reported as such
                    - metrics (): TODO
                - list (list[ConfigDict]): each element has the following keys:
                    - task (str): the name of the task to be evaluated
                    - freeze_featurizer (bool, default=False):
            - dataloader (ConfigDict): passed to the constructor of dataloaders
            - normalize_data (bool, default=False): whether to use the training set's mean and std to normalize all sets
            - early_stopping (ConfigDict):
            - lr_scheduler (ConfigDict): passed to the constructor of dataloaders
    """
    if not os.path.isdir(cfg.results_dir):
        os.makedirs(os.path.dirname(cfg.results_dir), exist_ok=True)

    datasets = make_datasets(cfg.tasks.common.get('datasets', {}), cfg.n_in, cfg.n_out)

    for Model, model_args in cfg.models:
        model = Model(n_in=cfg.n_in, n_features=cfg.n_features, space_dim=cfg.space_dim, **model_args)
        for task_cfg in cfg.tasks.list:
            task_datasets = datasets.copy()
            task_datasets.update(make_datasets(task_cfg.get('datasets', {}), cfg.n_in, cfg.n_out))

            train_model_for_task(model, task_cfg.task, task_datasets, cfg)

            if task_cfg.get('freeze_featurizer', False):
                model.freeze_featurizer()


def main(_argv):
    run_experiment(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
