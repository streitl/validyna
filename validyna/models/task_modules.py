from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch.optim import AdamW
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
        self.dataloaders = {name: DataLoader(self._transform_data(dataset), **cfg.dataloader, shuffle=True)
                            for name, dataset in datasets.items()}
        self.save_hyperparameters(ignore=['model', 'datasets'])

    @abstractmethod
    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        pass

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return [self.dataloaders['val'], self.dataloaders['test']]

    def _val_dataloader_name(self, dataloader_idx):
        return ['val', 'test'][dataloader_idx]

    def configure_optimizers(self):
        optimizers = [AdamW(self.parameters(), **self.cfg.optimizer)]
        schedulers = [{
            'scheduler': ReduceLROnPlateau(optimizers[0], **self.cfg.lr_scheduler, verbose=True),
            'monitor': f'{self.loss_name}.val',
        }]
        return optimizers, schedulers


class ChunkClassifier(ChunkModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, datasets: dict[str, ChunkMultiTaskDataset], cfg: ConfigDict):
        super().__init__(model, datasets, cfg, loss_name='loss.cross')
        model.prepare_to_classify(n_classes=datasets['train'].n_classes)

    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        return dataset.for_classification()

    def forward(self, x):
        out = self.model(x, kind='classify')
        if out.isnan().any() or out.isinf().any():
            print(f'Error in output of model (nan or inf)')
        return out

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

    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        return dataset.for_featurization()

    def forward(self, x):
        out = self.model(x, kind='featurize')
        if out.isnan().any() or out.isinf().any():
            print(f'Error in output of model (nan or inf)')
        return out

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

    def _transform_data(self, dataset: ChunkMultiTaskDataset) -> Dataset:
        return dataset.for_forecasting()

    def forward(self, x):
        out = self.model(x, kind='forecast')
        if out.isnan().any() or out.isinf().any():
            print(f'Error in output of model (nan or inf)')
        return out

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
