import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ecodyna.mutitask_models import MultiTaskTimeSeriesModel


class LightningClassifier(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.save_hyperparameters()

    def get_loss_acc(self, batch):
        x, y = batch
        out = self.model(x, kind='classification')
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(F.log_softmax(out), dim=1)
        acc = (y == preds).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.get_loss_acc(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.get_loss_acc(batch)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class LightningFeaturizer(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.save_hyperparameters()

    def get_loss(self, batch):
        a, p, n = batch
        f_a = self.model(a, kind='featurization')
        f_p = self.model(p, kind='featurization')
        f_n = self.model(n, kind='featurization')
        loss = F.triplet_margin_loss(f_a, f_p, f_n)  # TODO check
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class LightningForecaster(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.save_hyperparameters()
        self.prediction_func_name = 'forecast_in_chunks'  # TODO

    def get_loss(self, batch):
        x, y = batch
        out = self.model(x, kind='forecasting')
        loss = F.mse_loss(out, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (tensor,) = batch
        x = tensor[:, :self.model.n_in, :]
        predict = getattr(self.model, self.prediction_func_name)
        return predict(x, n=tensor.size(1) - self.model.n_in)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
