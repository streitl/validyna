import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW


# TODO
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
