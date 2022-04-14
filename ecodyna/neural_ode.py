import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torchdyn.core import NeuralODE


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
