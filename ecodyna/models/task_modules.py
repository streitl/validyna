import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ecodyna.models.mutitask_models import MultiTaskTimeSeriesModel


class ChunkClassifier(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])

    def get_loss_acc(self, batch):
        x, y = batch
        out = self.model(x, kind='classify')
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(F.log_softmax(out, dim=1), dim=1)
        acc = (y == preds).float().mean()
        return loss, acc

    # TODO check the effects of setting on_step and on_epoch inside self.log
    def training_step(self, batch, batch_idx):
        loss, acc = self.get_loss_acc(batch)
        self.log('loss', {'train': loss})
        self.log('acc', {'train': acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.get_loss_acc(batch)
        self.log('loss', {'val': loss})
        self.log('acc', {'val': acc})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (x,) = batch
        out = self.model(x, kind='classifier')
        preds = torch.argmax(F.log_softmax(out, dim=1), dim=1)
        return preds

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class ChunkTripletFeaturizer(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])

    def get_loss(self, batch):
        a, p, n = batch
        f_a = self.model(a, kind='featurize')
        f_p = self.model(p, kind='featurize')
        f_n = self.model(n, kind='featurize')
        loss = F.triplet_margin_loss(f_a, f_p, f_n)  # TODO check
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (x,) = batch
        return self.model(x, kind='featurize')

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class ChunkForecaster(pl.LightningModule):

    def __init__(self, model: MultiTaskTimeSeriesModel, lr: float = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])
        self.prediction_func = self.model.forecast_in_chunks

    def get_loss(self, batch):
        x, y = batch
        out = self.model(x, kind='forecast')
        loss = F.mse_loss(out, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (tensor,) = batch
        x = tensor[:, :self.model.n_in, :]
        predict = getattr(self.model, self.prediction_func_name)
        return predict(x, n=tensor.size(1) - self.model.n_in)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
