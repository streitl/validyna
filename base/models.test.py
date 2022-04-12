import unittest

import numpy as np
import pytorch_lightning as pl
import torch
from darts import TimeSeries
from darts.models import NBEATSModel, RNNModel
from dysts.flows import Lorenz

from data import build_in_out_pair_dataloader, generate_trajectories
from models import NBEATSForecaster, ForecasterLightning, LSTMForecaster


class TestModels(unittest.TestCase):
    def test_n_params_nbeats(self):
        trajectory_length = 100
        trajectory_count = 20

        n_in = 5
        n_out = 12
        n_stacks = 2
        n_blocks = 3
        n_layers = 4
        layer_widths = 32
        expansion_coefficient_dim = 5

        lorenz = Lorenz()
        data = generate_trajectories(chaos_model=lorenz,
                                     trajectory_length=trajectory_length,
                                     trajectory_count=trajectory_count,
                                     ic_fun=lambda: lorenz.ic + np.random.rand(lorenz.ic.shape) - 0.5 + lorenz.ic)

        dataset = torch.utils.data.TensorDataset(data)
        dataloader = build_in_out_pair_dataloader(dataset, n_in, n_out)
        ts = [TimeSeries.from_values(data[i]) for i in range(trajectory_count)]

        ours = NBEATSForecaster(n_features=len(lorenz.ic),
                                n_in=n_in,
                                n_out=n_out,
                                n_stacks=n_stacks,
                                n_blocks=n_blocks,
                                n_layers=n_layers,
                                layer_widths=layer_widths,
                                expansion_coefficient_dim=expansion_coefficient_dim)
        darts = NBEATSModel(input_chunk_length=n_in,
                            output_chunk_length=n_out,
                            num_stacks=n_stacks,
                            num_blocks=n_blocks,
                            num_layers=n_layers,
                            layer_widths=layer_widths,
                            expansion_coefficient_dim=expansion_coefficient_dim)

        trainer = pl.Trainer(max_epochs=1)
        darts.fit(series=ts, trainer=trainer)
        trainer.fit(ForecasterLightning(forecasting_model=ours), train_dataloaders=dataloader)

    def test_n_params_lstm(self):
        trajectory_length = 100
        trajectory_count = 20

        n_in = 5
        n_out = 12
        n_layers = 4
        n_hidden = 20

        lorenz = Lorenz()
        data = generate_trajectories(chaos_model=lorenz,
                                     trajectory_length=trajectory_length,
                                     trajectory_count=trajectory_count,
                                     ic_fun=lambda: np.random.rand(*lorenz.ic.shape) - 0.5 + lorenz.ic)

        dataset = torch.utils.data.TensorDataset(data)
        dataloader = build_in_out_pair_dataloader(dataset, n_in, n_out)
        ts = [TimeSeries.from_values(data[i]) for i in range(trajectory_count)]

        ours = LSTMForecaster(n_features=len(lorenz.ic),
                              n_hidden=n_hidden,
                              n_in=n_in,
                              n_out=n_out,
                              n_layers=n_layers)
        darts = RNNModel(input_chunk_length=n_in,
                         output_chunk_length=n_out,
                         model='LSTM',
                         hidden_dim=n_hidden,
                         n_rnn_layers=n_layers,
                         training_length=n_in)

        trainer = pl.Trainer(max_epochs=1)
        darts.fit(series=ts, trainer=trainer)
        trainer.fit(ForecasterLightning(forecasting_model=ours), train_dataloaders=dataloader)

    def test_prediction(self):
        trajectory_length = 100
        trajectory_count = 20

        n_in = 5
        n_out = 12
        n_layers = 4
        n_hidden = 20

        lorenz = Lorenz()
        data = generate_trajectories(chaos_model=lorenz,
                                     trajectory_length=trajectory_length,
                                     trajectory_count=trajectory_count,
                                     ic_fun=lambda: np.random.rand(*lorenz.ic.shape) - 0.5 + lorenz.ic)

        dataset = torch.utils.data.TensorDataset(data)
        dataloader = build_in_out_pair_dataloader(dataset, n_in, n_out)

        lstm = LSTMForecaster(n_features=len(lorenz.ic),
                              n_hidden=n_hidden,
                              n_in=n_in,
                              n_out=n_out,
                              n_layers=n_layers)

        trainer = pl.Trainer(max_epochs=1)
        forecaster = ForecasterLightning(forecasting_model=lstm)
        trainer.fit(forecaster, train_dataloaders=dataloader)

        print(lstm.forecast(dataset[0][0][:n_in].view(1, n_in, -1), n=trajectory_length-n_in))
        print(dataset[0])


if __name__ == '__main__':
    unittest.main()
