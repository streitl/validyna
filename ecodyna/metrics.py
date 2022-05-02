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

    def on_train_epoch_end(self, trainer: pl.Trainer, forecaster: LightningForecaster):
        model = forecaster.model
        metrics = {}
        for dataset_name, dataset in [('train', self.train_dataset), ('val', self.val_dataset)]:
            # Some boilerplate to access the inner tensor from the dataset
            (tensor,) = next(iter(DataLoader(dataset, batch_size=len(dataset))))
            B, T, D = tensor.size()
            start, target = tensor.split_with_sizes((model.n_in, T - model.n_in), dim=1)
            prediction = model.forecast_in_chunks(start, n=target.size(1))[:, model.n_in:, :]

            for metric_name, metric_func in [('full_mse', F.mse_loss)]:
                metric = metric_func(prediction, target).item()
                metrics[f'{dataset_name}_{metric_name}'] = metric

        trainer.logger.log_metrics(metrics)  # logging everything at once to avoid indexing by different steps


class RNNForecastMetricLogger(ForecastMetricLogger):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        super().__init__(train_dataset, val_dataset)

    def on_train_epoch_end(self, trainer: pl.Trainer, forecaster: LightningForecaster):
        if not isinstance(forecaster.model, MultiTaskRNN):
            raise ValueError(f'This Callback can only be applied to MultiTaskRNN')
        rnn: MultiTaskRNN = forecaster.model
        metrics = {}
        for dataset_name, dataset in [('train', self.train_dataset), ('val', self.val_dataset)]:
            # Some boilerplate to access the inner tensor from the dataset
            (tensor,) = next(iter(DataLoader(dataset, batch_size=len(dataset))))
            B, T, D = tensor.size()
            start, target = tensor.split_with_sizes((rnn.n_in, T - rnn.n_in), dim=1)
            for forecast_name, forecast_func in rnn.get_applicable_forecast_functions().items():
                prediction = forecast_func(start, n=target.size(1))[:, rnn.n_in:, :]

                for metric_name, metric_func in [('full_mse', F.mse_loss)]:
                    metric = metric_func(prediction, target).item()
                    metrics[f'{dataset_name}_{metric_name}_{forecast_name}'] = metric

        trainer.logger.log_metrics(metrics)  # logging everything at once to avoid indexing by different steps
