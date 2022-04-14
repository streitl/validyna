from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ecodyna.nbeats import NBEATS


def check_int_arg(arg, n_min, desc):
    if not isinstance(arg, int) or arg < n_min:
        raise ValueError(f'Integer {desc} must be >= {n_min} (got {arg})')
    return arg


class MultiTaskTimeSeriesModel(nn.Module, ABC):
    """
    TODO add description
    """

    def __init__(self, n_in: int, series_dim: int):
        super().__init__()

        # common to all tasks
        self.n_in = check_int_arg(n_in, 1, 'number of input time steps')
        self.series_dim = check_int_arg(series_dim, 1, 'time series dimension')

        # task-specific
        self.n_classes = None
        self.n_out = None
        self.n_features = None

    def is_prepared_for_classification(self) -> bool:
        return self.n_classes is not None

    def is_prepared_for_featurization(self) -> bool:
        return self.n_features is not None

    def is_prepared_for_forecasting(self) -> bool:
        return self.n_out is not None

    @abstractmethod
    def prepare_for_classification(self, n_classes):
        if self.is_prepared_for_classification():
            print(f'Warning: this {self.name()} is already prepared for classification')
        self.n_classes = check_int_arg(n_classes, 2, 'number of classes')

    @abstractmethod
    def prepare_for_featurization(self, n_features):
        if self.is_prepared_for_featurization():
            print(f'Warning: this {self.name()} is already prepared for featurization')
        self.n_features = check_int_arg(n_features, 1, 'number of features')

    @abstractmethod
    def prepare_for_forecasting(self, n_out):
        if self.is_prepared_for_forecasting():
            print(f'Warning: this {self.name()} is already prepared for forecasting')
        self.n_out = check_int_arg(n_out, 1, 'number of output time steps')

    def forward(self, x, kind: Literal['classification', 'featurization', 'forecasting']) -> Tensor:
        B, T, D = x.size()
        assert T == self.n_in, f'{self.name()} should take {self.n_in} time steps as input'
        if kind == 'classification' and self.is_prepared_for_classification():
            return self.forward_classify(x)
        elif kind == 'featurization' and self.is_prepared_for_featurization():
            return self.forward_featurize(x)
        elif kind == 'forecasting' and self.is_prepared_for_forecasting():
            return self.forward_forecast(x)
        else:
            raise ValueError(f'{self.name()} was not prepared for {kind}')

    @abstractmethod
    def forward_classify(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_featurize(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_forecast(self, x: Tensor) -> Tensor:
        pass

    def classify(self, x: Tensor) -> Tensor:
        y = self.forward_classify(x)
        y = F.log_softmax(y)
        return torch.argmax(y, dim=1)

    def featurize(self, x: Tensor) -> Tensor:
        return self.forward_featurize(x)

    def forecast_in_chunks(self, x: Tensor, n: int) -> Tensor:
        B, T, D = x.size()
        assert T == self.n_in, f'{self.name()} should take {self.n_in} time steps as input'
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x[:, :, :]
        for i in range(T, T + n, self.n_out):
            current_window = ts[:, i - self.n_in:i, :]
            out = self.forward_forecast(current_window)
            ts[:, i:i + self.n_out, :] = out[:, :min(self.n_out, T + n - i), :]  # don't go beyond n
        return ts

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass


class MultiTaskLSTM(MultiTaskTimeSeriesModel):

    def __init__(
            self,
            n_hidden: int,
            n_layers: int,
            n_in: int,
            series_dim: int,
            forecast_type: Literal['n_out', 'recurrent'] = 'n_out',
            *args, **kwargs
    ):
        super().__init__(n_in=n_in, series_dim=series_dim)
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=series_dim, hidden_size=n_hidden, num_layers=n_layers,
            *args, **kwargs
        )
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.forecast_type = forecast_type

        # LSTM is a natural featurizer
        # TODO should we use a featurizer layer to allow n_features != n_hidden?
        self.prepare_for_featurization(n_features=n_hidden)

        self.classifier = None
        self.forecaster_n_out = None
        self.forecaster_one = None

    def prepare_for_classification(self, n_classes):
        super().prepare_for_classification(n_classes)
        self.classifier = nn.Linear(self.n_features, self.n_classes)

    def prepare_for_forecasting(self, n_out):
        super().prepare_for_forecasting(n_out)
        self.forecaster_n_out = nn.Linear(self.n_features, self.series_dim * self.n_out)
        self.forecaster_one = nn.Linear(self.n_features, self.series_dim)

    def prepare_for_featurization(self, n_features):
        super().prepare_for_featurization(n_features)

    def forward_classify(self, x: Tensor) -> Tensor:
        features = self.forward_featurize(x)
        return self.classifier(features)

    def forward_featurize(self, x: Tensor) -> Tensor:
        output, last_hidden_layers = self.lstm(x)
        return output[:, -1, :]  # use the last hidden layer as features
        # TODO return self.featurizer(output[:, -1, :]) is more powerful? allows n_features != n_hidden

    def forward_forecast(self, x: Tensor) -> Tensor:
        if self.forecast_type == 'n_out':
            return self.forward_forecast_n_out(x)
        elif self.forecast_type == 'recurrent':
            return self.forward_forecast_recurrently(x)

    def forward_forecast_n_out(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        features = self.forward_featurize(x)
        return self.forecaster_n_out(features).view(B, self.n_out, self.series_dim)

    def forward_forecast_recurrently(self, x: Tensor) -> Tensor:
        return self.forecast_recurrently_one(x, n=self.n_out)[:, self.n_in:, :]

    def forecast_recurrently_one(self, x: Tensor, n: int) -> Tensor:
        B, T, D = x.size()
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x
        out, last_hidden_state = self.lstm(x)
        for i in range(T, T + n):
            ts[:, i, :] = self.forecaster_one(out[:, -1, :])
            out, last_hidden_state = self.lstm(ts[:, i:i + 1, :], last_hidden_state)
        return ts

    def forecast_recurrently_n_out_first(self, x: Tensor, n: int) -> Tensor:
        B, T, D = x.size()
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x
        out, last_hidden_state = self.lstm(x)
        for i in range(T, T + n):
            ts[:, i, :] = self.forecaster_n_out(out[:, -1, :]).view(B, self.n_out, D)[:, 0, :]
            out, last_hidden_state = self.lstm(ts[:, i:i + 1, :], last_hidden_state)
        return ts

    @staticmethod
    def name() -> str:
        return 'LSTM'


class MultiTaskNBEATS(MultiTaskTimeSeriesModel):

    def __init__(
            self,
            n_in: int,
            series_dim: int,
            n_out: int,
            *args, **kwargs
    ):
        super().__init__(n_in=n_in, series_dim=series_dim)

        self.nbeats = NBEATS(n_in=n_in * series_dim, n_out=n_out * series_dim, *args, **kwargs)

        # LSTM is a natural forecaster
        self.prepare_for_forecasting(n_out)

        self.classifier = None
        self.featurizer = None

    def prepare_for_featurization(self, n_features):
        super().prepare_for_featurization(n_features)
        self.featurizer = nn.Linear(self.n_out * self.series_dim, self.n_features)

    def prepare_for_classification(self, n_classes):
        if not self.is_prepared_for_featurization():
            raise ValueError(f'Prepare {self.name()} for featurization before preparing for classification')
        super().prepare_for_classification(n_classes)
        self.classifier = nn.Linear(self.n_features, self.n_classes)

    def prepare_for_forecasting(self, n_out):
        super().prepare_for_forecasting(n_out)

    def forward_classify(self, x: Tensor) -> Tensor:
        features = self.forward_featurize(x)
        return self.classifier(features)

    def forward_featurize(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        x = x.view(B, T * D)
        y = self.nbeats(x)
        return self.featurizer(y)

    def forward_forecast(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        x = x.view(B, T * D)
        y = self.nbeats(x)
        return y.view(B, -1, D)

    @staticmethod
    def name() -> str:
        return 'N-BEATS'
