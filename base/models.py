from abc import ABC, abstractmethod
from typing import Union, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import AdamW

"""
class PLNeuralODE(pl.LightningModule):

    def __init__(self, f: nn.Module):
        super().__init__()
        self.model = NeuralODE(f)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.model.nfe = 0
        pred = self(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        return {'loss': loss, 'progress_bar': {'nfe': self.model.nfe}}

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-6)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class TimeSeriesGAN(pl.LightningModule):
    class Generator(nn.Module):
        def __init__(self, series_dim, latent_dim, condition_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_size=latent_dim + condition_dim, hidden_size=hidden_dim)
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(hidden_dim, series_dim)
            )

        def forward(self, z, length=100):
            outs = []
            for t in range(length):
                _, (h_out, _) = self.model(z)
                outs.append(self.regressor(h_out))
            return torch.tensor(outs, requires_grad=True)

    class Discriminator(nn.Module):
        def __init__(self, series_dim, condition_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_size=series_dim + condition_dim, hidden_size=hidden_dim)

        def forward(self, ts, cond=None):
            l = ts
            if cond is not None:
                l = [torch.concat(x, cond) for x in ts]

            for i, x in enumerate(l):
                h = self.lstm(x)

    def __init__(self, series_dim, latent_dim, condition_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.series_dim = series_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.netG = self.Generator(
            series_dim=series_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim
        )
        self.netD = self.Discriminator(
            series_dim=series_dim,
            condition_dim=condition_dim,
            hidden_dim=condition_dim
        )

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-6)

    def training_step(self, batch, batch_idx):
        latent = torch.rand(batch.size(0), self.latent_dim)
"""


class Forecaster(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @abstractmethod
    def forecast(self, x: Tensor, n: int):
        pass

    @staticmethod
    @abstractmethod
    def name():
        pass


class ForecasterLightning(pl.LightningModule):

    def __init__(
            self,
            forecasting_model: Forecaster,
            criterion: nn.Module = nn.MSELoss(),
            lr: float = 1e-4,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = forecasting_model
        self.criterion = criterion
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


class NBEATS(nn.Module):
    class _Block(nn.Module):
        def __init__(
                self,
                n_in: int,
                n_out: int,
                n_layers: int,
                expansion_coefficient_dim: int,
                layer_width: int
        ):
            super().__init__()
            self.n_in = n_in
            self.n_out = n_out
            self.n_layers = n_layers
            self.expansion_coefficient_dim = expansion_coefficient_dim
            self.layer_widths = layer_width

            self.FC_stack = nn.ModuleList(
                [nn.Linear(n_in, layer_width)] + [nn.Linear(layer_width, layer_width) for _ in range(n_layers - 1)]
            )
            self.FC_backcast = nn.Linear(layer_width, expansion_coefficient_dim)
            self.FC_forecast = nn.Linear(layer_width, expansion_coefficient_dim)

            self.g_backcast = nn.Linear(expansion_coefficient_dim, n_in)
            self.g_forecast = nn.Linear(expansion_coefficient_dim, n_out)

        def forward(self, x: Tensor):
            for layer in self.FC_stack:
                x = F.relu(layer(x))

            backcast = self.g_backcast(self.FC_backcast(x))
            forecast = self.g_forecast(self.FC_forecast(x))
            return backcast, forecast

    class _Stack(nn.Module):
        def __init__(self, n_in: int, n_out: int, n_blocks: int, *args, **kwargs):
            super().__init__()
            self.n_in = n_in
            self.n_out = n_out
            self.n_blocks = n_blocks

            self.blocks = nn.ModuleList([NBEATS._Block(n_in, n_out, *args, **kwargs) for i in range(n_blocks)])

        def forward(self, x: Tensor):
            B, T = x.size()
            assert T == self.n_in, f'NBeats Stack should take {self.n_in} time steps as input'
            forecast = torch.zeros(B, self.n_out)
            for block in self.blocks:
                block_backcast, block_forecast = block(x)
                x = x - block_backcast
                forecast += block_forecast
            return x, forecast

    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_stacks: int = 4,
            n_blocks: int = 4,
            n_layers: int = 4,
            expansion_coefficient_dim: int = 5,
            layer_widths: Union[int, List[int]] = 256
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.expansion_coefficient_dim = expansion_coefficient_dim
        self.layer_widths = layer_widths
        if isinstance(layer_widths, int):
            self.layer_widths = [layer_widths] * n_stacks

        self.stacks = nn.ModuleList([
            NBEATS._Stack(
                n_in=n_in,
                n_out=n_out,
                n_blocks=n_blocks,
                n_layers=n_layers,
                layer_width=self.layer_widths[i],
                expansion_coefficient_dim=expansion_coefficient_dim
            )
            for i in range(n_stacks)
        ])
        self.stacks[-1].blocks[-1].FC_backcast.requires_grad_(False)
        self.stacks[-1].blocks[-1].g_backcast.requires_grad_(False)

    def forward(self, x: Tensor):
        B, T = x.size()
        assert T == self.n_in, f'NBeats should take {self.n_in} time steps as input'
        backcast = x
        forecast = torch.zeros(B, self.n_out)
        for stack in self.stacks:
            backcast, stack_forecast = stack(backcast)
            forecast += stack_forecast
        return forecast


class NBEATSForecaster(Forecaster):

    def __init__(self, n_in: int, n_out: int, n_features: int, *args, **kwargs):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_features = n_features

        self.nbeats = NBEATS(n_in=n_in * n_features, n_out=n_out * n_features, *args, **kwargs)

    @staticmethod
    def name():
        return 'N-BEATS'

    def forward(self, x: Tensor):
        B, T, D = x.size()
        assert T == self.n_in, f'N-BEATS should take {self.n_in} time steps as input'
        # Flattening and de-flattening trick to make N-BEATS work with multidimensional data
        x = x.view(B, T * D)
        y = self.nbeats(x)
        return y.view(B, -1, D)

    def forecast(self, x: Tensor, n: int):
        B, T, D = x.size()
        assert T == self.n_in, f'N-BEATS should take {self.n_in} time steps as input'
        ts = torch.empty((B, T + n, D))
        ts[:, :T, :] = x
        for i in range(T, T + n, self.n_out):
            out = self(ts[:, i - self.n_in:i, :])
            ts[:, i:i + self.n_out, :] = out[:, :min(self.n_out, T + n - i), :]
        return ts


class LSTMForecaster(Forecaster):

    def __init__(self, n_in: int, n_out: int, n_features: int, n_hidden: int, n_layers: int, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            batch_first=True, input_size=n_features, hidden_size=n_hidden, num_layers=n_layers,
            *args, **kwargs
        )
        self.regressor = nn.Linear(n_hidden, n_features * n_out)
        self.n_in = n_in
        self.n_out = n_out
        self.n_features = n_features

    @staticmethod
    def name():
        return 'LSTM'

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        assert T == self.n_in, f'LSTM should take {self.n_in} time steps as input'
        out, last_hidden_state = self.lstm(x)
        # regress on the last hidden layer
        return self.regressor(out[:, -1, :]).view(B, self.n_out, self.n_features)

    def forecast(self, x: Tensor, n: int, strict_n_in: bool = True) -> Tensor:
        B, T, D = x.size()
        assert T == self.n_in, f'LSTM should take {self.n_in} time steps as input'
        ts = torch.empty((B, T + n, D))
        ts[:, :T, :] = x
        if strict_n_in:
            for i in range(T, T + n, self.n_out):
                out = self(ts[:, i - self.n_in:i, :])
                ts[:, i:i + self.n_out, :] = out[:, :min(self.n_out, T + n - i), :]
        else:
            out, last_hidden_state = self.lstm(x)
            for i in range(T, T + n):
                ts[:, i, :] = self.regressor(out[:, -1, :]).view(B, self.n_out, self.n_features)[:, -1, :]
                out, last_hidden_state = self.lstm(ts[:, i:i + 1, :], last_hidden_state)
        return ts
