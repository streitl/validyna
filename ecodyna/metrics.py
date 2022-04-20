from abc import abstractmethod, ABC

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ecodyna.mutitask_models import MultiTaskRNN
from ecodyna.pl_wrappers import LightningForecaster


class DatasetMetricLogger(pl.Callback, ABC):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    @abstractmethod
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass


class ForecastMetricLogger(DatasetMetricLogger):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        super().__init__(train_dataset, val_dataset)

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
                prediction = forecaster.forecast_in_chunks(tensor[:, :forecaster.n_in, :], n=T - forecaster.n_in)
                mse += F.mse_loss(prediction, tensor)
                break  # not needed but to clarify that the for loop only exists to extract data from dataloader
            trainer.logger.log_metrics({metric_name: mse})


class RNNForecastMetricLogger(ForecastMetricLogger):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        super().__init__(train_dataset, val_dataset)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightningForecaster):
        if not isinstance(pl_module.model, MultiTaskRNN):
            raise ValueError(f'This Callback can only be applied to MultiTaskRNNs')
        forecaster: MultiTaskRNN = pl_module.model
        metrics = {}
        for metric_name, dataset in [
            ('train_traj_mse', self.train_dataset),
            ('val_traj_mse', self.val_dataset)
        ]:
            dataloader = DataLoader(dataset, batch_size=len(dataset))
            for func_name, forecast_func in [
                ('one', forecaster.forecast_recurrently_one),
                ('chunks', forecaster.forecast_in_chunks),
                ('first', forecaster.forecast_recurrently_n_out_first)
            ]:
                mse = 0
                for (tensor,) in dataloader:
                    B, T, D = tensor.size()
                    prediction = forecast_func(tensor[:, :forecaster.n_in, :], n=T - forecaster.n_in)
                    mse += F.mse_loss(prediction, tensor)
                    break  # not needed but to clarify that the for loop only exists to extract data from dataloader
                metrics[f'{metric_name}_{func_name}'] = mse

        trainer.logger.log_metrics(metrics)  # logging everything at once to avoid indexing by different steps
