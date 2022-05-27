from itertools import groupby

import dysts.base
import dysts.flows
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import ROOT_DIR
from ecodyna.data import ChunkClassDataset, load_from_params
from ecodyna.models.task_modules import ChunkClassifier
from ecodyna.tasks.common import experiment_setup


def run_classification_of_attractors_experiment(params: dict):
    train_size, val_size = experiment_setup(**params)

    used_attractors = [getattr(dysts.flows, attractor_name)() for attractor_name in params['experiment']['attractors']]
    used_attractors.sort(key=lambda x: len(x.ic))  # itertools.groupby needs sorted data
    attractors_per_dim = groupby(used_attractors, key=lambda x: len(x.ic))

    for space_dim, attractors in attractors_per_dim:
        if space_dim != 3:
            continue
        print(f'Loading trajectories for attractors of dimension {space_dim}')
        datasets = {attractor.name: TensorDataset(load_from_params(attractor=attractor.name, **params['data']))
                    for attractor in tqdm(list(attractors))}

        dataset = ChunkClassDataset(datasets, train_size, val_size, **params['in_out'])
        train_ds, val_ds = dataset.random_split()
        train_dl = DataLoader(train_ds, **params['dataloader'], shuffle=True)
        val_dl = DataLoader(val_ds, **params['dataloader'], shuffle=True)

        for Model, model_params in params['models']['list']:
            model = Model(space_dim=space_dim, n_classes=dataset.n_classes,
                          **model_params, **params['models']['common'])
            
            classifier = ChunkClassifier(model=model)

            run_id = f'{model.name()}_dim_{space_dim}'
            wandb_logger = WandbLogger(
                save_dir=f'{ROOT_DIR}/results',
                project=params['experiment']['project'],
                name=run_id,
                id=run_id,
                config={
                    'ml': model.hyperparams,
                    'data': params['data'],
                    'dataloader': params['dataloader'],
                    'experiment': params['experiment'],
                    'trainer': {k: f'{v}' for k, v in params['trainer'].items()}
                }
            )

            trainer = pl.Trainer(
                logger=wandb_logger, **params['trainer'],
                callbacks=[EarlyStopping('loss.val', patience=5, check_on_train_epoch_end=True),
                           LearningRateMonitor()]
            )
            trainer.fit(classifier, train_dataloaders=train_dl, val_dataloaders=val_dl)

            wandb_logger.experiment.finish(quiet=True)
