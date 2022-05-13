from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.optim import AdamW


# TODO
class TimeSeriesGAN(pl.LightningModule):
    class Generator(nn.Module):
        def __init__(self, space_dim, latent_dim, condition_dim, n_hidden):
            super().__init__()
            self.lstm = nn.LSTM(batch_first=True, input_size=latent_dim + condition_dim, hidden_size=n_hidden)
            self.regressor = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(n_hidden, space_dim)
            )

        def forward(self, z, length=100):
            outs = []
            for t in range(length):
                _, (h_out, _) = self.model(z)
                outs.append(self.regressor(h_out))
            return torch.tensor(outs, requires_grad=True)

    class Discriminator(nn.Module):
        def __init__(self, space_dim, condition_dim, n_hidden):
            super().__init__()
            self.lstm = nn.LSTM(batch_first=True, input_size=space_dim + condition_dim, hidden_size=n_hidden)
            self.classifier = nn.Linear(n_hidden, 2)

        def forward(self, x: Tensor, cond: Optional[Tensor] = None):
            if cond is not None:
                x = torch.concat((x, cond))
            output, last_hidden_layer = self.lstm(x)
            return self.classifier(output[: -1, :])

    def __init__(self, space_dim: int, latent_dim: int, condition_dim: int, n_hidden: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.space_dim = space_dim
        self.condition_dim = condition_dim
        self.hidden_dim = space_dim
        self.netG = self.Generator(
            space_dim=space_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            n_hidden=n_hidden
        )
        self.netD = self.Discriminator(
            space_dim=space_dim,
            condition_dim=condition_dim,
            n_hidden=n_hidden
        )

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-6)

    def training_step(self, batch, batch_idx):
        latent = torch.rand(batch.size(0), self.latent_dim)
