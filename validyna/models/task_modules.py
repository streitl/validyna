from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset

from validyna.data import ChunkMultiTaskDataset
from validyna.models.multitask_models import MultiTaskTimeSeriesModel


class ChunkModule(pl.LightningModule, ABC):
    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.datasets = datasets
        if cfg.get('normalize_data', default=False):
            train_mean, train_std = datasets['train'].mean, datasets['train'].std
            for dataset in self.datasets.values():
                dataset.normalize(train_mean, train_std)
        self.dataloaders = {
            name: DataLoader(self._transform_data(dataset), **cfg.dataloader)
            for name, dataset in datasets.items()
        }
        self.val_dataloader_names = ['val'] + [n for n in self.dataloaders.keys() if n not in ['train', 'val']]
        self.save_hyperparameters(ignore=['model', 'datasets'])

    @abstractmethod
    def _transform_data(self, dataset: ChunkMultiTaskDataset):
        pass

    @abstractmethod
    def loss_name(self) -> str:
        pass

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return [self.dataloaders[name] for name in self.val_dataloader_names]

    def val_dataloader_name(self, dataloader_idx):
        return self.val_dataloader_names[dataloader_idx]

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
    def get_metrics(self, batch) -> dict[str, float]:
        pass

    def log_metrics(self, metrics: dict[str, float], set_name: str) -> None:
        for name, value in metrics.items():
            if name == 'loss':
                name = self.loss_name()
            self.log(f'{name}.{set_name}', value, add_dataloader_idx=False)

    def training_step(self, batch, batch_idx=0):
        metrics = self.get_metrics(batch)
        self.log_metrics(metrics, set_name='train')
        return metrics

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        self.log_metrics(self.get_metrics(batch), set_name=self.val_dataloader_name(dataloader_idx))


class ChunkClassifier(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)
        model.prepare_to_classify(n_classes=datasets['train'].n_classes)

    def loss_name(self) -> str:
        return 'loss.cross'

    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        return dataset.for_classification()

    def forward(self, x):
        return self.model(x, kind='classify')

    def get_metrics(self, batch):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(F.log_softmax(out, dim=1), dim=1)
        acc = (y == preds).float().mean()
        return {'loss': loss, 'acc': acc}

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(F.log_softmax(self(x), dim=1), dim=1)


class ChunkTripletFeaturizer(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)
        model.prepare_to_featurize(n_features=cfg.n_features)

    def loss_name(self) -> str:
        return 'loss.triplet'

    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        return dataset.for_featurization()

    def forward(self, x):
        return self.model(x, kind='featurize')

    def get_metrics(self, batch):
        a, p, n = batch
        loss = F.triplet_margin_loss(self(a), self(p), self(n))
        return {'loss': loss}

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        (x,) = batch
        return self(x)


class ChunkForecaster(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg)
        model.prepare_to_forecast(n_out=cfg.n_out)

    def loss_name(self) -> str:
        return 'loss.mse'

    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        return dataset.for_forecasting()

    def forward(self, x):
        return self.model(x, kind='forecast')

    def get_metrics(self, batch):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        return {'loss': loss}

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        (tensor,) = batch
        x = tensor[:, :self.model.n_in, :]
        return self.model.forecast_in_chunks(x, n=tensor.size(1) - self.model.n_in)
