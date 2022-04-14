import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from ecodyna.pl_wrappers import LightningForecaster


class ForecastMetricLogger(pl.Callback):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightningForecaster):
        forecaster = pl_module.model
        for metric_name, dataset in [
            ('train_traj_mse', self.train_dataset),
            ('val_traj_mse', self.val_dataset)
        ]:
            dataloader = DataLoader(dataset, batch_size=len(dataset))
            mse = 0
            for (tensor,) in dataloader:
                B, T, D = tensor.size()
                prediction = forecaster.forecast_in_chunks(tensor[:, :forecaster.n_in, :], n=T-forecaster.n_in)
                mse += F.mse_loss(prediction, tensor)
            trainer.logger.log_metrics({metric_name: mse})
