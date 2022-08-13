import os
from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from absl import app
from ml_collections import ConfigDict, config_flags
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from validyna.data import SliceMultiTaskDataset
from validyna.models import multitask_models as mm
from validyna.registry import task_registry, model_registry, metric_registry

_CONFIG = config_flags.DEFINE_config_file('cfg')


def run_model_training(
        model: mm.MultiTaskTimeSeriesModel,
        datasets: dict[str, SliceMultiTaskDataset],
        cfg: ConfigDict,
        run_cfg: Optional[ConfigDict] = None,
):
    """
    This method takes care of training the given model on the given datasets, using the configuration specified in cfg
    (the global configuration) and run_cfg (the configuration specific to this run).

    This method allows training the model for different tasks, to load weights from a previous run, to log metrics and
    model checkpoints on the cloud using Weights and Biases, and to control the training procedure by specifying the
    use of early stopping, model hyper-parameters, and more.

    Args:
        - model (MultiTaskTimeSeriesModel): the model to be trained
        - datasets (dict of SliceMultiTaskDatasets): dict with at least the training and validation sets
        - cfg (ConfigDict): the global configuration as specified in <run_experiment>, including the keys:
            - seed (int): the random seed for reproducibility
            - trainer (dict): the parameters to be passed to the Pytorch-Lightning trainer object
            - task (str): the task the model should be trained on (e.g. 'classification')
            - use_wandb (bool, default=False): whether to use Weights and Biases to log metrics and save models
            - project (str): name of the project
            - run_suffix (str): a text to add to the name of the run after the name of the model
            - results_dir (str): where to save the results, namely metrics and model checkpoints
            - early_stopping (optional dict): the parameters of the EarlyStopping callback or None for no early stopping
        - run_cfg (ConfigDict): the configuration specific to this run
            - task (str): the task the model should be trained on (e.g. 'classification')
            - run_suffix (str): a text to add to the name of the run after the name of the model
            - restore_run_suffix (optional str): the name of a previous run to be loaded into the model
            - trainer_callbacks (list[Tuple[str, dict, str]]): list of triplets where the first is the name of the
                | callback (as registered in metrics registry), the second is the parameters to be passed to it,
                | and the third is the task where the callback should be used (e.g. 'classification', 'all', ...)
    """
    pl.seed_everything(cfg.seed, workers=True)
    trainer_kwargs = {k: v for k, v in cfg.trainer.items() if k != 'callbacks'}
    task = cfg.get('task') or run_cfg.task
    # When Weights and Biases is used, we need to determine the run name, create the logger, and save/load models
    if cfg.get('use_wandb', default=False):
        run_suffix = f"{cfg.get('run_suffix')}_{run_cfg.get('run_suffix')}" \
            if cfg.get('run_suffix') and run_cfg.get('run_suffix') \
            else cfg.get('run_suffix') or run_cfg.get('run_suffix')
        run_name = f'{model.name()}_{run_suffix or task}'
        wandb_logger = WandbLogger(project=cfg.project, name=run_name, id=run_name, save_dir=cfg.results_dir,
                                   log_model=True, config=dict(model=model.hyperparams, **cfg))
        # Allows restoring a model from a previous run
        if 'restore_run_suffix' in run_cfg:
            restore_run_name = f'{model.name()}_{run_cfg.restore_run_suffix}'
            model_at = wandb.run.use_artifact(f'{cfg.project}.{restore_run_name}:latest')
            model_dir = model_at.download(os.path.join(cfg.results_dir, cfg.project, restore_run_name))
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model.h5')))
        trainer_kwargs['logger'] = wandb_logger
    module = task_registry[task](model=model, datasets=datasets, cfg=cfg)
    trainer_callbacks = [metric_registry[name](**kwargs)
                         for name, kwargs, t in cfg.trainer.get('callbacks', []) + run_cfg.get('trainer_callbacks', [])
                         if t == task or t == 'all']
    trainer_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if 'early_stopping' in cfg:
        trainer_callbacks += [EarlyStopping(monitor=f'{module.loss_name()}.val', **cfg.early_stopping)]

    trainer = pl.Trainer(callbacks=trainer_callbacks, default_root_dir=cfg.results_dir, **trainer_kwargs)
    trainer.fit(module)

    if cfg.get('use_wandb', default=False):
        model_artifact = wandb.Artifact(f'{cfg.project}.{run_name}', type='model')
        dir_path = os.path.join(cfg.results_dir, cfg.project, run_name)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_path, 'model.h5'))
        model_artifact.add_dir(dir_path)
        wandb.run.log_artifact(model_artifact)
        trainer_kwargs['logger'].experiment.finish(quiet=True)


def run_experiment(cfg: ConfigDict):
    """
    This method takes a configuration and runs the experiment defined by it.
    Then, for each different run in the experiment, it calls <run_model_training>

    Args:
        - cfg (ConfigDict): a configuration dictionary with the following keys:
            - results_dir (str): the path of the directory to save the results
            - datasets (Callable[[], [dict]]: a callable function returning a dictionary of MultiTaskDatasets
            - models (list[Tuple[str, dict]]): tuples where the first is a model name registered in the model registry,
                | and the second is a dictionary with the parameters to be passed to the model constructor
            - runs (list[ConfigDict]): for each ordered run, specify extra configuration; allowed keys:
                - datasets: same structure as cfg.datasets
                - fresh_model (bool, default=False): whether to use a fresh model instead of the one from previous run
                - freeze_featurizer (bool, default=False): whether to freeze the featurizer weights after this run
                > more keys shown in the documentation of <run_model_training>
            - n_in (int): number of time steps that are given to the models
            - n_out (int): (if forecasting is involved) number of future time steps predicted by the model
            - n_features (int): the number of features used by the model to encode a trajectory slice
            - space_dim (int): the space dimension (vector size at each time step) of the used attractors
            > more keys specified in the documentation of <run_model_training> and of <data.SliceMultiTaskDataset>
    """
    if not os.path.isdir(cfg.results_dir):
        os.makedirs(cfg.results_dir, exist_ok=True)

    datasets = cfg.get('datasets', default=lambda: {})()

    # Iterate over all models
    for model_name, model_args in cfg.models:
        Model = model_registry[model_name]
        model = Model(n_in=cfg.n_in, n_features=cfg.n_features, space_dim=cfg.space_dim, **model_args)
        # Iterate over runs for each model
        for run_cfg in cfg.runs:
            run_datasets = {**datasets}
            run_datasets.update(run_cfg.get('datasets', default=lambda: {})())
            run_datasets = {k: SliceMultiTaskDataset(v, cfg.n_in, cfg.n_out) for k, v in run_datasets.items()}
            serializable_cfg = ConfigDict({k: v for k, v in cfg.items() if k not in ['datasets', 'runs']})

            if run_cfg.get('fresh_model', default=False):
                model = Model(n_in=cfg.n_in, n_features=cfg.n_features, space_dim=cfg.space_dim, **model_args)

            run_model_training(model=model, datasets=run_datasets, run_cfg=run_cfg, cfg=serializable_cfg)

            if run_cfg.get('freeze_featurizer', default=False):
                model.freeze_featurizer()


def main(_argv):
    run_experiment(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
