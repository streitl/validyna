import os
from typing import Optional

import pytorch_lightning as pl
from absl import app
from ml_collections import ConfigDict, config_flags
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from validyna.data import ChunkMultiTaskDataset
from validyna.models import multitask_models as mm
from validyna.registry import task_registry, model_registry, metric_registry

_CONFIG = config_flags.DEFINE_config_file('cfg')


def run_model_training(
        model: mm.MultiTaskTimeSeriesModel,
        datasets: dict[str, ChunkMultiTaskDataset],
        cfg: ConfigDict,
        run_cfg: Optional[ConfigDict] = None,
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
    task = cfg.get('task') or run_cfg.task
    if cfg.get('use_wandb', default=False):
        run_suffix = f"{cfg.get('run_suffix')}_{run_cfg.get('run_suffix')}"\
            if cfg.get('run_suffix') and run_cfg.get('run_suffix') \
            else cfg.get('run_suffix') or run_cfg.get('run_suffix')
        run_name = f'{model.name()}_{run_suffix or task}'
        wandb_logger = WandbLogger(project=cfg.project, name=run_name, id=run_name, save_dir=cfg.results_dir,
                                   config=dict(model=model.hyperparams, **cfg))
        trainer_kwargs['logger'] = wandb_logger
    module = task_registry[task](model=model, datasets=datasets, cfg=cfg)
    trainer_callbacks = [metric_registry[name](**kwargs)
                         for name, kwargs, t in cfg.trainer.get('callbacks', []) + run_cfg.get('trainer_callbacks', [])
                         if t == task or t == 'all']
    trainer_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if 'early_stopping' in cfg:
        trainer_callbacks += [EarlyStopping(monitor=f'{module.loss_name()}.val', **cfg.early_stopping)]
    trainer = pl.Trainer(callbacks=trainer_callbacks, **trainer_kwargs)
    trainer.fit(module)
    if cfg.get('use_wandb', default=False):
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
                        - train (dict[str, ChunkMultiTaskDataset]): data used to optimize the model
                        - val (dict[str, ChunkMultiTaskDataset]): data used for early stopping, learning rate adjustment
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
    print('main start')
    if not os.path.isdir(cfg.results_dir):
        os.makedirs(cfg.results_dir, exist_ok=True)

    datasets = cfg.get('datasets', default=lambda: {})()

    for model_name, model_args in cfg.models:
        Model = model_registry[model_name]
        model = Model(n_in=cfg.n_in, n_features=cfg.n_features, space_dim=cfg.space_dim, **model_args)
        for run_cfg in cfg.runs:
            task_datasets = {**datasets}
            task_datasets.update(run_cfg.get('datasets', default=lambda: {})())

            run_model_training(model=model, datasets=task_datasets, run_cfg=run_cfg,
                               cfg=ConfigDict({k: v for k, v in cfg.items() if k not in ['datasets', 'runs']}))

            if run_cfg.get('freeze_featurizer', default=False):
                model.freeze_featurizer()


def main(_argv):
    run_experiment(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
