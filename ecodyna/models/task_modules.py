import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ecodyna.models.mutitask_models import MultiTaskTimeSeriesModel


class ChunkClassifier(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-2, patience: int = 1, factor: float = 0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_params = {'lr': lr}
        self.lr_scheduler_params = {'patience': patience, 'factor': factor}
        self.save_hyperparameters(ignore=['model'])
        self.loss = 'loss.cross'

    def forward(self, x):
        return self.model(x, kind='classify')

    def get_loss_acc(self, batch):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(F.log_softmax(out, dim=1), dim=1)
        acc = (y == preds).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.get_loss_acc(batch)
        self.log(f'{self.loss}.train', loss)
        self.log('acc.train', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.get_loss_acc(batch)
        self.log(f'{self.loss}.val', loss)
        self.log('acc.val', acc)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (x,) = batch
        preds = torch.argmax(F.log_softmax(self(x), dim=1), dim=1)
        return preds

    def configure_optimizers(self):
        optimizers = [AdamW(self.parameters(), **self.optimizer_params)]
        schedulers = [{
            'scheduler': ReduceLROnPlateau(optimizers[0], **self.lr_scheduler_params, verbose=True),
            'monitor': f'{self.loss}.val',
        }]
        return optimizers, schedulers


class ChunkTripletFeaturizer(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-2, patience: int = 1, factor: float = 0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_params = {'lr': lr}
        self.lr_scheduler_params = {'patience': patience, 'factor': factor}
        self.save_hyperparameters(ignore=['model'])
        self.loss = 'loss.triplet'

    def forward(self, x):
        return self.model(x, kind='featurize')

    def get_loss(self, batch):
        a, p, n = batch
        loss = F.triplet_margin_loss(self(a), self(p), self(n))  # TODO check
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f'{self.loss}.train', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f'{self.loss}.val', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (x,) = batch
        return self(x)

    def configure_optimizers(self):
        optimizers = [AdamW(self.parameters(), **self.optimizer_params)]
        schedulers = [{
            'scheduler': ReduceLROnPlateau(optimizers[0], **self.lr_scheduler_params, verbose=True),
            'monitor': f'{self.loss}.val',
        }]
        return optimizers, schedulers


class ChunkForecaster(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-2, patience: int = 1, factor: float = 0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_params = {'lr': lr}
        self.lr_scheduler_params = {'patience': patience, 'factor': factor}
        self.save_hyperparameters(ignore=['model'])
        self.loss = 'loss.mse'

    def forward(self, x):
        return self.model(x, kind='forecast')

    def get_loss(self, batch):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f'{self.loss}.train', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f'{self.loss}.val', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO test that the prediction function is correctly configured
        (tensor,) = batch
        x = tensor[:, :self.model.n_in, :]
        return self.prediction_func(x, n=tensor.size(1) - self.model.n_in)

    def configure_optimizers(self):
        optimizers = [AdamW(self.parameters(), **self.optimizer_params)]
        schedulers = [{
            'scheduler': ReduceLROnPlateau(optimizers[0], **self.lr_scheduler_params, verbose=True),
            'monitor': f'{self.loss}.val',
        }]
        return optimizers, schedulers
