from abc import ABC, abstractmethod
from typing import Optional

import pdb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from validyna.data import SliceMultiTaskDataset
from validyna.models.multitask_models import MultiTaskTimeSeriesModel


class SliceModule(pl.LightningModule, ABC):
    """
    Args:
        - model: an instance of a MultiTaskTimeSeriesModel
        - datasets: each key is a set name including 'train' and 'val', with its associated MultiTaskDataset
        - cfg: a ConfigDict as specified in <main.run_experiment>, with the following extra keys:
            - scale_data (bool, default=False): whether to scale data in [-1, 1] per-component for each attractor
            - dataloader (ConfigDict): passed to the constructor of dataloaders
            - optimizer (Tuple[str, dict]): left: name of a torch optimizer; right: optimizer parameters
            - lr_scheduler (Tuple[str, dict]): left: name of a torch lr_scheduler; right: scheduler parameters
    """
    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, SliceMultiTaskDataset], cfg: ConfigDict):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.datasets = datasets
        # Scaling
        mins, maxs = None, None
        if cfg.scale_data:
            mins, maxs = datasets['train'].get_mins_maxs()
        for name in datasets.keys():
            datasets[name].process_data(mins, maxs)

        self.dataloaders = {
            name: DataLoader(dataset.tensor_dataset(), **cfg.dataloader)
            for name, dataset in datasets.items()
        }
        self.val_dataloader_names = ['val'] + [n for n in self.dataloaders.keys() if n not in ['train', 'val']]
        self.save_hyperparameters(ignore=['model', 'datasets'])

    @classmethod
    @abstractmethod
    def loss_name(cls) -> str:
        pass

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return [self.dataloaders[name] for name in self.val_dataloader_names]

    def configure_optimizers(self):
        optim_name, optim_params = self.cfg.optimizer
        Optim = getattr(torch.optim, optim_name)
        optimizers = [Optim(self.parameters(), **optim_params)]
        schedulers = []
        if 'lr_scheduler' in self.cfg:
            scheduler_name, scheduler_params = self.cfg.lr_scheduler
            LRScheduler = getattr(torch.optim.lr_scheduler, scheduler_name)
            schedulers.append({
                'scheduler': LRScheduler(optimizers[0], **scheduler_params),
                'monitor': f'{self.loss_name()}.val',
            })
        return optimizers, schedulers

    @abstractmethod
    def get_metrics(self, batch, set_name: Optional[str]) -> dict[str, float]:
        pass

    def log_metrics(self, metrics: dict[str, float], set_name: str) -> None:
        for name, value in metrics.items():
            if name == 'loss':
                name = self.loss_name()
            # print(f'{name}.{set_name}', value)
            self.log(f'{name}.{set_name}', value, add_dataloader_idx=False)

    def training_step(self, batch, batch_idx=0):
        set_name = 'train'
        metrics = self.get_metrics(batch, set_name)
        self.log_metrics(metrics, set_name)
        return metrics

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        set_name = self.val_dataloader_names[dataloader_idx]
        metrics = self.get_metrics(batch, set_name)
        self.log_metrics(metrics, set_name)
        return metrics


class SliceClassifier(SliceModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, SliceMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)
        model.prepare_to_classify(n_classes=datasets['train'].n_classes)

    @classmethod
    def loss_name(cls) -> str:
        return 'loss.cross'

    def forward(self, x):
        return self.model(x, kind='classify')

    def get_metrics(self, batch, set_name: str):
        x_in, x_out, x_class = batch
        out = self(x_in)
        loss = F.cross_entropy(out, x_class)
        preds = torch.argmax(F.log_softmax(out, dim=1), dim=1)
        acc = (x_class == preds).float().mean()
        # pdb.set_trace()
        return {'loss': loss, 'acc': acc}

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        x = batch[0]
        return torch.argmax(F.log_softmax(self(x), dim=1), dim=1)


class SliceFeaturizer(SliceModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, SliceMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)
        model.prepare_to_featurize(n_features=cfg.n_features)

    @classmethod
    def loss_name(cls) -> str:
        return 'loss.triplet'

    def forward(self, x):
        return self.model(x, kind='featurize')

    def get_metrics(self, batch, set_name: str):
        x_in, x_out, x_class = batch
        a = x_in
        p, n = self.datasets[set_name].get_positive_negative_batch(x_class)
        loss = F.triplet_margin_loss(self(a), self(p.to(self.device)), self(n.to(self.device)))
        return {'loss': loss}

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        x = batch[0]
        return self(x)


class SliceForecaster(SliceModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, SliceMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)
        model.prepare_to_forecast(n_out=cfg.n_out)

    @classmethod
    def loss_name(cls) -> str:
        return 'loss.mse'

    def forward(self, x):
        return self.model(x, kind='forecast')

    def get_metrics(self, batch, set_name=None):
        x_in, x_out, x_class = batch
        loss = F.mse_loss(self(x_in), x_out)
        return {'loss': loss}

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        tensor = batch[0]
        x = tensor[:, :self.model.n_in, :]
        return self.model.forecast_by_slices(x, n=tensor.size(1) - self.model.n_in)


class SliceHybrid(SliceModule):
    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, SliceMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)

        model.prepare_to_featurize(n_features=cfg.n_features)
        model.prepare_to_forecast(n_out=cfg.n_out)
        model.prepare_to_classify(n_classes=datasets['train'].n_classes)

        assert sum(cfg.loss_coefficients.values()) == 1, 'sum of coefficients must be 1'
        self.loss_scales = {'cross': 0.3, 'mse': 0.005, 'triplet': 0.12}
        self.loss_coefficients = {k: v / self.loss_scales[k] for k, v in cfg.loss_coefficients.items()}

    @classmethod
    def loss_name(cls) -> str:
        return 'loss.total'

    def get_metrics(self, batch, set_name: str):
        metrics = dict()
        losses = dict()

        for cls in [SliceClassifier, SliceFeaturizer, SliceForecaster]:
            self.__class__ = cls
            metrics.update(self.get_metrics(batch, set_name))
            self.__class__ = SliceHybrid
            loss = metrics.pop('loss')
            metrics[cls.loss_name()] = loss
            losses[cls.loss_name().replace('loss.', '')] = loss
        loss = sum([self.loss_coefficients[n] * losses[n] for n in self.loss_coefficients.keys()])

        return {'loss': loss, **metrics}
