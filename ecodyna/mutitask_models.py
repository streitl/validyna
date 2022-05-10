from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ecodyna.nbeats import NBEATS


def check_int_arg(arg: any, n_min: int, desc: str):
    """
    Small helper function that checks whether the given argument is an integer with value at least n_min,
    and shows a message using the given description when it is not the case.
    """
    if not isinstance(arg, int) or arg < n_min:
        raise ValueError(f'Integer {desc} must be >= {n_min} (got {arg})')
    return arg


def make_simple_mlp(i: int, h: int, o: int, n_hidden_layers: int = 3, activation: nn.Module = nn.ReLU()):
    """
    Args:
        - i: number of input units
        - h: number of units in the hidden layers
        - o: number of output units
        - n_hidden_layers: number of hidden layers
        - activation: activation function applied after each linear layer except the last
    """
    modules = [nn.Linear(i, h), activation] + [nn.Linear(h, h), activation] * n_hidden_layers + [nn.Linear(h, o)]
    return nn.Sequential(*modules)


class MultiTaskTimeSeriesModel(nn.Module, ABC):
    """
    Base class for models that can be used to classify, featurize, and forecast time series.

    Throughout this class and its subclasses, we define:
    - B as the batch size
    - T as the sequence length (number of time steps)
    - D as the dimension of each time point

    Note that all current subclasses use `batch_first=True` in PyTorch modules.
    """

    def __init__(
            self,
            n_in: int,
            space_dim: int,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            n_out: Optional[int] = None
    ):
        super().__init__()
        assert (n_classes or n_features or n_out) is not None, \
            'One of `n_classes`, `n_features`, `n_out` must be non-None'

        # common to all tasks
        self.n_in = check_int_arg(n_in, n_min=1, desc='number of input time steps')
        self.space_dim = check_int_arg(space_dim, n_min=1, desc='space dimension')

        # task-specific
        self.n_classes = None
        self.n_out = None
        self.n_features = None

        self.hyperparams = {}
        self.register_hyperparams(n_in=n_in, space_dim=space_dim,
                                  n_classes=n_classes, n_features=n_features, n_out=n_out)

    def register_hyperparams(self, **kwargs):
        self.hyperparams.update(**kwargs)

    @property
    @abstractmethod
    def _natural_n_features(self):
        """
        The natural number of features created by the model
        """
        pass

    """
    These 3 methods should be overwritten and called before the model is used for the corresponding task.
    The overrider should call the corresponding method in super()
    """

    @abstractmethod
    def prepare_to_classify(self, n_classes: int):
        if self.is_prepared_to_classify():
            print(f'Warning: this {self.name()} is already prepared to classify')
        self.n_classes = check_int_arg(n_classes, n_min=2, desc='number of classes')

    @abstractmethod
    def prepare_to_featurize(self, n_features: int):
        if self.is_prepared_to_featurize():
            print(f'Warning: this {self.name()} is already prepared to featurize')
        self.n_features = check_int_arg(n_features, n_min=1, desc='number of features')

    @abstractmethod
    def prepare_to_forecast(self, n_out: int):
        if self.is_prepared_to_forecast():
            print(f'Warning: this {self.name()} is already prepared to forecast')
        self.n_out = check_int_arg(n_out, n_min=1, desc='number of output time steps')

    """
    We use n_classes, n_features and n_out as markers of whether the model is prepared for the corresponding task.
    """

    def is_prepared_to_classify(self) -> bool:
        return self.n_classes is not None

    def is_prepared_to_featurize(self) -> bool:
        return self.n_features is not None

    def is_prepared_to_forecast(self) -> bool:
        return self.n_out is not None

    """
    The methods with the actual logic for the forward pass of each task.
    """

    @abstractmethod
    def _forward_classify(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _forward_featurize(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _forward_forecast(self, x: Tensor) -> Tensor:
        pass

    def forward(self, x, kind: Literal['classify', 'featurize', 'forecast']) -> Tensor:
        """
        A wrapper for the forward methods of all tasks.
        Checks whether the model is prepared for the given task and if the model inputs and outputs are well-formed.
        """
        B, T, D = x.size()
        assert T == self.n_in, f'{self.name()} should take {self.n_in} time steps as input'
        assert D == self.space_dim, f'{self.name()} should take inputs with dimension {self.space_dim} as input'
        if kind == 'classify' and self.is_prepared_to_classify():
            class_outputs = self._forward_classify(x)
            assert class_outputs.size(1) == self.n_classes
            return class_outputs
        elif kind == 'featurize' and self.is_prepared_to_featurize():
            features = self._forward_featurize(x)
            assert features.size(1) == self.n_features
            return features
        elif kind == 'forecast' and self.is_prepared_to_forecast():
            forecast = self._forward_forecast(x)
            assert forecast.size(1) == self.n_out
            assert forecast.size(2) == self.space_dim
            return forecast
        else:
            raise ValueError(f'{self.name()} was not prepared to {kind}')

    """
    The public methods to be used for prediction in the corresponding tasks.
    They wrap around the corresponding forward methods and make sure that the model is prepared for the task.
    """

    def classify(self, x: Tensor) -> Tensor:
        if not self.is_prepared_to_classify():
            raise ValueError(f'{self.name()} was not prepared to classify')
        y = self._forward_classify(x)
        y = F.log_softmax(y)
        return torch.argmax(y, dim=1)

    def featurize(self, x: Tensor) -> Tensor:
        if not self.is_prepared_to_featurize():
            raise ValueError(f'{self.name()} was not prepared to featurize')
        return self._forward_featurize(x)

    def forecast_in_chunks(self, x: Tensor, n: int) -> Tensor:
        """
        Inspired from the implementation in Darts.
        To forecast n future time steps where n != n_out,
        """
        if not self.is_prepared_to_forecast():
            raise ValueError(f'{self.name()} was not prepared to forecast')
        B, T, D = x.size()
        assert T == self.n_in, f'{self.name()} should take {self.n_in} time steps as input'
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x[:, :, :]
        for i in range(T, T + n, self.n_out):
            current_window = ts[:, i - self.n_in:i, :]
            out = self._forward_forecast(current_window)
            ts[:, i:i + self.n_out, :] = out[:, :min(self.n_out, T + n - i), :]  # don't go beyond n
        return ts

    """
    These methods are used to allow freezing the weights of the featurizer part of the model in order to enable
    transfer-learning across tasks.
    """

    @abstractmethod
    def _get_featurizer_parameters(self) -> list[nn.Parameter]:
        pass

    def freeze_featurizer(self):
        for parameter in self._get_featurizer_parameters():
            parameter.requires_grad_(False)

    def unfreeze_featurizer(self):
        for parameter in self._get_featurizer_parameters():
            parameter.requires_grad_(True)

    """
    The name of the model for pretty-printing.
    """

    @abstractmethod
    def name(self) -> str:
        pass


class MyRNN(MultiTaskTimeSeriesModel):
    """
    Implements LSTM and GRU.
    RNNs are natural featurizers.
    """
    _natural_n_features = None

    def __init__(
            self,
            n_in: int,
            space_dim: int,
            model: Literal['GRU', 'LSTM'],
            n_layers: int,
            n_hidden: int,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            n_out: Optional[int] = None,
            classifier: Optional[nn.Module] = None,
            featurizer: Optional[nn.Module] = None,
            forecaster: Optional[nn.Module] = None,
            forecast_type: Literal['multi', 'one_by_one'] = 'one_by_one',
            **kwargs
    ):
        self._natural_n_features = n_hidden

        assert model in ['GRU', 'LSTM'], 'Only GRU and LSTM are supported'
        assert forecast_type in ['multi', 'one_by_one'], '`forecast_type^ must be `multi` or `one_by_one`'

        if n_features is None:
            n_features = self._natural_n_features

        super().__init__(n_in=n_in, space_dim=space_dim, n_classes=n_classes, n_features=n_features, n_out=n_out)
        self.register_hyperparams(model=model, n_layers=n_layers, forecast_type=forecast_type, **kwargs)

        self.model = model
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.forecast_type = forecast_type

        self.rnn = getattr(nn, model)(batch_first=True, input_size=space_dim, hidden_size=n_hidden,
                                      num_layers=n_layers, **kwargs)

        self.classifier = None
        self.featurizer = None
        self.forecaster = None

        self.prepare_to_featurize(n_features=n_features, featurizer=featurizer)
        if n_classes is not None:
            self.prepare_to_classify(n_classes=n_classes, classifier=classifier)
        if n_out is not None:
            self.prepare_to_forecast(n_out=n_out, forecaster=forecaster)

    # Preparation methods
    def prepare_to_classify(self, n_classes: int, classifier: Optional[nn.Module] = None):
        if not self.is_prepared_to_featurize():
            raise ValueError(f'Prepare {self.name()} to featurize before preparing to classify')
        super().prepare_to_classify(n_classes=n_classes)
        self.classifier = classifier or make_simple_mlp(i=self.n_features, h=self.n_features, o=n_classes)

    def prepare_to_featurize(self, n_features: int, featurizer: Optional[nn.Module] = None):
        super().prepare_to_featurize(n_features=n_features)
        self.featurizer = featurizer or nn.Linear(self._natural_n_features, self.n_features)

    def prepare_to_forecast(self, n_out: int, forecaster: Optional[nn.Module] = None):
        if not self.is_prepared_to_featurize():
            raise ValueError(f'Prepare {self.name()} to featurize before preparing to classify')
        super().prepare_to_forecast(n_out=n_out)
        true_n_out = self.space_dim * (1 if self.forecast_type == 'one_by_one' else self.n_out)
        self.forecaster = forecaster or make_simple_mlp(i=self.n_features, h=self.n_features, o=true_n_out)

    # Forward methods
    def _forward_classify(self, x: Tensor) -> Tensor:
        features = self._forward_featurize(x)
        return self.classifier(features)

    def _forward_featurize(self, x: Tensor) -> Tensor:
        output, last_hidden_layers = self.rnn(x)
        natural_features = output[:, -1, :]  # use the last hidden layer as natural features
        features = self.featurizer(natural_features)
        return features

    def _forward_forecast(self, x: Tensor) -> Tensor:
        if self.forecast_type == 'multi':
            return self.forecast_multi_all(x)
        elif self.forecast_type == 'one_by_one':
            return self.forecast_recurrently_one_by_one(x, n=self.n_out)[:, self.n_in:, :]
        else:
            raise ValueError(f'Unknown RNN forecast type: {self.forecast_type}')

    # RNNs allow additional kinds of forecasting
    def forecast_multi_all(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        features = self._forward_featurize(x)
        return self.forecaster(features).reshape(B, self.n_out, self.space_dim)

    def forecast_recurrently_one_by_one(self, x: Tensor, n: int) -> Tensor:
        assert self.forecast_type == 'one_by_one', 'This forecast function requires forecast type `one_by_one`'
        B, T, D = x.size()
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x
        out, last_hidden_state = self.rnn(x)
        for i in range(T, T + n):
            ts[:, i, :] = self.forecaster(out[:, -1, :])
            out, last_hidden_state = self.rnn(ts[:, i:i + 1, :], last_hidden_state)
        return ts

    def forecast_recurrently_multi_first(self, x: Tensor, n: int) -> Tensor:
        """
        Forecasts recurrently by only keeping the first prediction of the multi-timestep output.
        """
        assert self.forecast_type == 'multi', 'This forecast function requires forecast type `multi`'
        B, T, D = x.size()
        ts = torch.empty((B, T + n, D), dtype=x.dtype)
        ts[:, :T, :] = x
        out, last_hidden_state = self.rnn(x)
        for i in range(T, T + n):
            ts[:, i, :] = self.forecaster(out[:, -1, :]).reshape(B, self.n_out, D)[:, 0, :]
            out, last_hidden_state = self.rnn(ts[:, i:i + 1, :], last_hidden_state)
        return ts

    def get_applicable_forecast_functions(self):
        """
        Returns a dictionary containing all the forecasting functions that can be used by this model
        given its forecasting type.
        """
        functions = {'chunks': self.forecast_in_chunks}
        if self.forecast_type == 'one_by_one':
            functions['one_by_one'] = self.forecast_recurrently_one_by_one
        elif self.forecast_type == 'multi':
            functions['multi'] = self.forecast_recurrently_chunk_first
        return functions

    # Overriding of the other methods
    def _get_featurizer_parameters(self):
        return self.rnn.parameters()

    def name(self) -> str:
        return self.model


class MyNBEATS(MultiTaskTimeSeriesModel):
    """
    Implements N-BEATS (Neural Basis Expansion Analysis for interpretable Time Series forecasting).

    N-BEATS is a natural forecaster, but it relies on features in its neural basis expansion,
    so it is also a natural featurizer.
    """

    _natural_n_features = None

    def __init__(
            self,
            n_in: int,
            space_dim: int,
            n_blocks: int,
            n_stacks: int,
            expansion_coefficient_dim: int,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            n_out: Optional[int] = None,
            classifier: Optional[nn.Module] = None,
            featurizer: Optional[nn.Module] = None,
            **kwargs
    ):
        self._natural_n_features = n_blocks * n_stacks * expansion_coefficient_dim

        if (n_features and n_out) is None:
            n_features = self._natural_n_features

        super().__init__(n_in=n_in, space_dim=space_dim, n_classes=n_classes, n_features=n_features, n_out=n_out)
        self.register_hyperparams(n_blocks=n_blocks, n_stacks=n_stacks,
                                  expansion_coefficient_dim=expansion_coefficient_dim, **kwargs)

        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.expansion_coefficient_dim = expansion_coefficient_dim

        flattened_n_out = n_out * space_dim if n_out is not None else None
        self.nbeats = NBEATS(n_in=n_in * space_dim, n_out=flattened_n_out, n_blocks=n_blocks,
                             n_stacks=n_stacks, expansion_coefficient_dim=expansion_coefficient_dim, **kwargs)

        self.classifier = None
        self.featurizer = None

        if n_features is not None:
            self.prepare_to_featurize(n_features=n_features, featurizer=featurizer)
        if n_out is not None:
            self.prepare_to_forecast(n_out=n_out)
        if n_classes is not None:
            self.prepare_to_classify(n_classes=n_classes, classifier=classifier)

    # Preparation methods
    def prepare_to_classify(self, n_classes: int, classifier: Optional[nn.Module] = None):
        if not self.is_prepared_to_featurize():
            raise ValueError(f'Prepare {self.name()} to featurize before preparing to classify')
        super().prepare_to_classify(n_classes=n_classes)
        self.classifier = classifier or make_simple_mlp(i=self.n_features, h=self.n_features, o=self.n_classes)

    def prepare_to_featurize(self, n_features: int, featurizer: Optional[nn.Module] = None):
        super().prepare_to_featurize(n_features=n_features)
        self.featurizer = featurizer or nn.Linear(self._natural_n_features, self.n_features)

    def prepare_to_forecast(self, n_out: int):
        super().prepare_to_forecast(n_out=n_out)
        self.nbeats.set_n_out(n_out * self.space_dim)

    # Forward methods
    def _forward_classify(self, x: Tensor) -> Tensor:
        features = self._forward_featurize(x)
        return self.classifier(features)

    def _forward_featurize(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        x = x.reshape(B, T * D)
        natural_features = self.nbeats.featurize(x)
        features = self.featurizer(natural_features)
        return features

    def _forward_forecast(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        # N-BEATS is built for uni-variate time series, so we must flatten our multiple variables
        x = x.reshape(B, T * D)
        y = self.nbeats(x)
        return y.reshape(B, -1, D)

    # Overriding of the other methods
    def _get_featurizer_parameters(self):
        # All the parameters except those of block.g_forward
        modules = [module for stack in self.nbeats.stacks
                   for block in stack
                   for module in [block.FC_stack, block.FC_forward, block.FC_backward, block.g_backward]]
        return [param for module in modules for param in module.parameters()]

    def name(self) -> str:
        return 'N-BEATS'


class MyTransformer(MultiTaskTimeSeriesModel):
    """
    Implements a simplified Transformer model (only actually uses the encoder).
    The Transformer is built for seq2seq. but it is a natural featurizer.
    """

    _natural_n_features = None

    def __init__(
            self,
            n_in: int,
            space_dim: int,
            n_classes: Optional[int] = None,
            n_features: Optional[int] = None,
            n_out: Optional[int] = None,
            classifier: Optional[nn.Module] = None,
            featurizer: Optional[nn.Module] = None,
            forecaster: Optional[nn.Module] = None,
            **kwargs
    ):
        self._natural_n_features = n_in * space_dim

        if n_features is None:
            n_features = self._natural_n_features

        super().__init__(n_in=n_in, space_dim=space_dim, n_classes=n_classes, n_features=n_features, n_out=n_out)
        self.register_hyperparams(**kwargs)

        # We instantiate an entire transformer even though we only use the encoder part
        self.transformer = nn.Transformer(d_model=space_dim, batch_first=True, **kwargs)

        self.classifier = None
        self.featurizer = None
        self.forecaster = None

        if n_features is not None:
            self.prepare_to_featurize(n_features=n_features, featurizer=featurizer)
        if n_classes is not None:
            self.prepare_to_classify(n_classes=n_classes, classifier=classifier)
        if n_out is not None:
            self.prepare_to_forecast(n_out=n_out, forecaster=forecaster)

    # Preparation methods
    def prepare_to_classify(self, n_classes: int, classifier: Optional[nn.Module] = None):
        if not self.is_prepared_to_featurize():
            raise ValueError(f'Prepare {self.name()} to featurize before preparing to classify')
        super().prepare_to_classify(n_classes=n_classes)
        self.classifier = classifier or make_simple_mlp(i=self.n_features, h=self.n_features, o=self.n_classes)

    def prepare_to_featurize(self, n_features: int, featurizer: Optional[nn.Module] = None):
        super().prepare_to_featurize(n_features=n_features)
        self.featurizer = featurizer or nn.Linear(self._natural_n_features, self.n_features)

    def prepare_to_forecast(self, n_out: int, forecaster: Optional[nn.Module] = None):
        if not self.is_prepared_to_featurize():
            raise ValueError(f'Prepare {self.name()} to featurize before preparing to classify')
        super().prepare_to_forecast(n_out)
        self.forecaster = forecaster or make_simple_mlp(i=self.n_features, h=self.n_features,
                                                        o=self.n_out * self.space_dim)

    # Forward methods
    def _forward_classify(self, x: Tensor) -> Tensor:
        features = self._forward_featurize(x)
        return self.classifier(features)

    def _forward_featurize(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        natural_features = self.transformer.encoder(x).reshape(B, T * D)
        features = self.featurizer(natural_features)
        return features

    def _forward_forecast(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        features = self._forward_featurize(x)
        return self.forecaster(features).reshape(B, self.n_out, D)

    # Overriding of the other methods
    def _get_featurizer_parameters(self):
        return self.transformer.encoder.parameters()

    def name(self) -> str:
        return 'Transformer'
