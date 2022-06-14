from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from validyna.data import ChunkMultiTaskDataset
from validyna.models.multitask_models import MultiTaskTimeSeriesModel


class ChunkModule(pl.LightningModule, ABC):
    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict,
                 loss_name: str):
        super().__init__()
        self.model = model
        self.loss_name = loss_name
        self.cfg = cfg
        data_params = dict()
        if cfg.get('normalize_data', False):
            data_params = dict(mean=datasets['train'].mean, std=datasets['train'].std)
        self.dataloaders = {
            name: DataLoader(self._transform_data(dataset, **data_params), shuffle=True, **cfg.dataloader)
            for name, dataset in datasets.items()
        }
        self.dataloader_names = ['val'] + [n for n in self.dataloaders.keys() if n not in ['train', 'val']]
        self.save_hyperparameters(ignore=['model', 'datasets'])

    @abstractmethod
    def _transform_data(self, dataset: ChunkMultiTaskDataset, **kwargs) -> Dataset:
        pass

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return [self.dataloaders[name] for name in self.dataloader_names]

    def _val_dataloader_name(self, dataloader_idx):
        return self.dataloader_names[dataloader_idx]

    def configure_optimizers(self):
        Optim, optim_params = self.cfg.optimizer
        optimizers = [Optim(self.parameters(), **optim_params)]
        schedulers = []
        if 'lr_scheduler' in self.cfg:
            Scheduler, scheduler_params = self.cfg.lr_scheduler
            schedulers.append({
                'scheduler': Scheduler(optimizers[0], **scheduler_params),
                'monitor': f'{self.loss_name}.val',
            })
        return optimizers, schedulers


class ChunkClassifier(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg, loss_name='loss.cross')
        model.prepare_to_classify(n_classes=datasets['train'].n_classes)

    def _transform_data(self, dataset: ChunkMultiTaskDataset, **kwargs) -> Dataset:
        return dataset.for_classification(**kwargs)

    def forward(self, x):
        return self.model(x, kind='classify')

    def _get_loss_acc(self, batch):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(F.log_softmax(out, dim=1), dim=1)
        acc = (y == preds).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._get_loss_acc(batch)
        self.log(f'{self.loss_name}.train', loss)
        self.log('acc.train', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self._get_loss_acc(batch)
        which = self._val_dataloader_name(dataloader_idx)
        self.log(f'{self.loss_name}.{which}', loss, add_dataloader_idx=False)
        self.log(f'acc.{which}', acc, add_dataloader_idx=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (x,) = batch
        return torch.argmax(F.log_softmax(self(x), dim=1), dim=1)


class ChunkTripletFeaturizer(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg, loss_name='loss.triplet')
        model.prepare_to_featurize(n_features=cfg.n_features)

    def _transform_data(self, dataset: ChunkMultiTaskDataset, **kwargs) -> Dataset:
        return dataset.for_featurization(**kwargs)

    def forward(self, x):
        return self.model(x, kind='featurize')

    def _get_loss(self, batch):
        a, p, n = batch
        return F.triplet_margin_loss(self(a), self(p), self(n))

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log(f'{self.loss_name}.train', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss = self._get_loss(batch)
        which = self._val_dataloader_name(dataloader_idx)
        self.log(f'{self.loss_name}.{which}', loss, add_dataloader_idx=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (x,) = batch
        return self(x)


class ChunkForecaster(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg, loss_name='loss.mse')
        model.prepare_to_forecast(n_out=cfg.n_out)

    def _transform_data(self, dataset: ChunkMultiTaskDataset, **kwargs) -> Dataset:
        return dataset.for_forecasting(**kwargs)

    def forward(self, x):
        return self.model(x, kind='forecast')

    def _get_loss(self, batch):
        x, y = batch
        return F.mse_loss(self(x), y)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log(f'{self.loss_name}.train', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss = self._get_loss(batch)
        which = self._val_dataloader_name(dataloader_idx)
        self.log(f'{self.loss_name}.{which}', loss, add_dataloader_idx=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (tensor,) = batch
        x = tensor[:, :self.model.n_in, :]
        return self.model.forecast_in_chunks(x, n=tensor.size(1) - self.model.n_in)
