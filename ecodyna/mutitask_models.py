from abc import ABC, abstractmethod
from typing import Literal, Optional

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
    Base class for models that can be used for classification, featurization, and forecasting.
    """

    def __init__(self, n_in: int, space_dim: int):
        super().__init__()

        # common to all tasks
        self.n_in = check_int_arg(n_in, 1, 'number of input time steps')
        self.space_dim = check_int_arg(space_dim, 1, 'space dimension')

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
    def prepare_for_classification(self, n_classes: int):
        """
        Should be overwritten and called before the model is used for classification.
        The overrider should call super().prepare_for_classification
        """
        if self.is_prepared_for_classification():
            print(f'Warning: this {self.name()} is already prepared for classification')
        self.n_classes = check_int_arg(n_classes, 2, 'number of classes')

    @abstractmethod
    def prepare_for_featurization(self, n_features: int):
        """
        Should be overwritten and called before the model is used for featurization.
        The overrider should call super().prepare_for_featurization
        """
        if self.is_prepared_for_featurization():
            print(f'Warning: this {self.name()} is already prepared for featurization')
        self.n_features = check_int_arg(n_features, 1, 'number of features')

    @abstractmethod
    def prepare_for_forecasting(self, n_out: int):
        """
        Should be overwritten and called before the model is used for forecasting.
        The overrider should call super().prepare_for_forecasting
        """
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
        if not self.is_prepared_for_classification():
            raise ValueError(f'{self.name()} was not prepared for classification')
        y = self.forward_classify(x)
        y = F.log_softmax(y)
        return torch.argmax(y, dim=1)

    def featurize(self, x: Tensor) -> Tensor:
        if not self.is_prepared_for_featurization():
            raise ValueError(f'{self.name()} was not prepared for featurization')
        return self.forward_featurize(x)

    def forecast_in_chunks(self, x: Tensor, n: int) -> Tensor:
        if not self.is_prepared_for_forecasting():
            raise ValueError(f'{self.name()} was not prepared for forecasting')
        B, T, D = x.size()
        assert T == self.n_in, f'{self.name()} should take {self.n_in} time steps as input'
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x[:, :, :]
        for i in range(T, T + n, self.n_out):
            current_window = ts[:, i - self.n_in:i, :]
            out = self.forward_forecast(current_window)
            ts[:, i:i + self.n_out, :] = out[:, :min(self.n_out, T + n - i), :]  # don't go beyond n
        return ts

    @abstractmethod
    def name(self) -> str:
        pass


class MultiTaskRNN(MultiTaskTimeSeriesModel):
    """
    Implements LSTM and GRU.

    TODO remove forecasting types of move them to another class
    """

    def __init__(
            self,
            model: Literal['GRU', 'LSTM'],
            n_layers: int,
            n_in: int,
            space_dim: int,
            n_hidden: Optional[int] = None,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            n_out: Optional[int] = None,
            classifier: Optional[nn.Module] = None,
            forecaster: Optional[nn.Module] = None,
            forecast_type: Literal['n_out', 'recurrent'] = 'recurrent',
            *args, **kwargs
    ):
        super().__init__(n_in=n_in, space_dim=space_dim)

        assert model in ['GRU', 'LSTM'], 'Only GRU and LSTM are supported'
        assert not (n_features is None and n_hidden is None), 'Must specify the number of hidden units or features'
        assert n_features is None or n_hidden is None or n_hidden == n_features, \
            f'The current {self.name()} only accepts n_features == n_hidden'
        n_features = n_features or n_hidden
        n_hidden = n_features

        self.model = model
        self.rnn = getattr(nn, model)(
            batch_first=True,
            input_size=space_dim, hidden_size=n_hidden, num_layers=n_layers,
            *args, **kwargs
        )
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.forecast_type = forecast_type

        self.classifier = None
        self.forecaster_n_out = None
        self.forecaster_one = None

        # RNNs are natural featurizers
        # TODO should we use a featurizer layer to allow n_features != n_hidden?
        self.prepare_for_featurization(n_features=n_features)

        if n_classes is not None:
            self.prepare_for_classification(n_classes=n_classes, classifier=classifier)

        if n_out is not None:
            self.prepare_for_forecasting(n_out=n_out, forecaster=forecaster)

    def prepare_for_classification(self, n_classes: int, classifier: nn.Module = None):
        super().prepare_for_classification(n_classes=n_classes)
        if classifier is None:
            classifier = nn.Linear(self.n_features, self.n_classes)
        self.classifier = classifier

    def prepare_for_forecasting(self, n_out: int, forecaster: nn.Module = None):
        super().prepare_for_forecasting(n_out=n_out)
        self.forecaster_n_out = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, self.space_dim * self.n_out)
        )
        if forecaster is None:
            forecaster = nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.space_dim)
            )
        self.forecaster_one = forecaster

    def prepare_for_featurization(self, n_features: int):
        super().prepare_for_featurization(n_features=n_features)

    def forward_classify(self, x: Tensor) -> Tensor:
        features = self.forward_featurize(x)
        return self.classifier(features)

    def forward_featurize(self, x: Tensor) -> Tensor:
        output, last_hidden_layers = self.rnn(x)
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
        return self.forecaster_n_out(features).view(B, self.n_out, self.space_dim)

    def forward_forecast_recurrently(self, x: Tensor) -> Tensor:
        return self.forecast_recurrently_one(x, n=self.n_out)[:, self.n_in:, :]

    def forecast_recurrently_one(self, x: Tensor, n: int) -> Tensor:
        B, T, D = x.size()
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x
        out, last_hidden_state = self.rnn(x)
        for i in range(T, T + n):
            ts[:, i, :] = self.forecaster_one(out[:, -1, :])
            out, last_hidden_state = self.rnn(ts[:, i:i + 1, :], last_hidden_state)
        return ts

    def forecast_recurrently_n_out_first(self, x: Tensor, n: int) -> Tensor:
        B, T, D = x.size()
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x
        out, last_hidden_state = self.rnn(x)
        for i in range(T, T + n):
            ts[:, i, :] = self.forecaster_n_out(out[:, -1, :]).view(B, self.n_out, D)[:, 0, :]
            out, last_hidden_state = self.rnn(ts[:, i:i + 1, :], last_hidden_state)
        return ts

    def name(self) -> str:
        return self.model


class MultiTaskNBEATS(MultiTaskTimeSeriesModel):
    """
    Implements N-BEATS (Neural Basis Expansion Analysis for interpretable Time Series forecasting).
    """

    def __init__(
            self,
            n_in: int,
            space_dim: int,
            n_out: int,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            *args, **kwargs
    ):
        super().__init__(n_in=n_in, space_dim=space_dim)

        self.nbeats = NBEATS(n_in=n_in * space_dim, n_out=n_out * space_dim, *args, **kwargs)

        # N-BEATS is a natural forecaster
        self.prepare_for_forecasting(n_out)

        self.classifier = None
        self.featurizer = None

        # Prepare for featurization BEFORE classification if both are needed
        if n_features is not None:
            self.prepare_for_featurization(n_features=n_features)
        if n_classes is not None:
            self.prepare_for_classification(n_classes=n_classes)

    def prepare_for_classification(self, n_classes: int):
        if not self.is_prepared_for_featurization():
            raise ValueError(f'Prepare {self.name()} for featurization before preparing for classification')
        super().prepare_for_classification(n_classes=n_classes)
        self.classifier = nn.Linear(self.n_features, self.n_classes)

    def prepare_for_forecasting(self, n_out: int):
        super().prepare_for_forecasting(n_out=n_out)

    def prepare_for_featurization(self, n_features: int):
        super().prepare_for_featurization(n_features=n_features)
        self.featurizer = nn.Linear(self.n_out * self.space_dim, self.n_features)

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

    def name(self) -> str:
        return 'N-BEATS'


class MultiTaskTransformer(MultiTaskTimeSeriesModel):

    def __init__(
            self,
            n_in: int,
            space_dim: int,
            n_out: int,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            *args, **kwargs
    ):
        super().__init__(n_in=n_in, space_dim=space_dim)

        self.transformer = nn.Transformer(batch_first=True, *args, **kwargs)

        # Transformer is designed for seq2seq which is easily adaptable to forecasting
        self.prepare_for_forecasting(n_out=n_out)
        # TODO

    def prepare_for_classification(self, n_classes: int):
        # TODO
        pass

    def prepare_for_featurization(self, n_features: int):
        # TODO
        pass

    def prepare_for_forecasting(self, n_out: int):
        # TODO
        pass

    def forward_classify(self, x: Tensor) -> Tensor:
        # TODO
        pass

    def forward_featurize(self, x: Tensor) -> Tensor:
        # TODO
        pass

    def forward_forecast(self, x: Tensor) -> Tensor:
        # TODO
        y = x[:, self.n_in:, :]
        x = x[:, :self.n_in, :]
        return self.transformer(x, y)

    def name(self) -> str:
        return 'Transformer'
