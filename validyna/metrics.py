from abc import ABC, abstractmethod
from typing import Tuple, Type, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader

from validyna.models.task_modules import ChunkClassifier, ChunkModule, ChunkForecaster, ChunkTripletFeaturizer


class Prober(pl.Callback):
    def __init__(self, cls: Type[ChunkModule]):
        self.cls = cls
        self.prober = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if isinstance(pl_module, self.cls):
            raise ValueError("The prober should be for a task different than the training one.")
        self.prober = self.cls(pl_module.model, pl_module.datasets, pl_module.cfg)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, unused: int = 0):
        self.prober.training_step(self.prober.train_dataloader()[batch_idx], batch_idx=batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int):
        self.prober.validation_step(self.prober.val_dataloader()[dataloader_idx][batch_idx],
                                    batch_idx=batch_idx,
                                    dataloader_idx=dataloader_idx)


class ClassMetricLogger(pl.Callback, ABC):

    def __init__(self, class_of_interest: str):
        self.class_of_interest = class_of_interest
        self.class_index = None

    @abstractmethod
    def required_module_type(self) -> Type[ChunkModule]:
        pass

    @abstractmethod
    def get_metrics(self, pl_module: pl.LightningModule, batch: Any) -> dict[str, float]:
        pass

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(pl_module, ChunkModule):
            raise ValueError(f'This callback only works for {ChunkModule.__name__}')
        if not isinstance(pl_module, self.required_module_type()):
            raise ValueError(
                f'Can only log the metric {self.__class__.__name__} for class {self.required_module_type().__name__}')
        self.class_index = pl_module.datasets['train'].classes[self.class_of_interest]


class ClassSensitivitySpecificity(ClassMetricLogger):

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return ChunkClassifier

    def get_metrics(self, classifier: ChunkClassifier, batch: Tuple[Tensor, Tensor]):
        x, target = batch
        out = classifier.predict_step(batch)
        # Ignore all different classes and only consider this class or any other
        target = target == self.class_index
        out = out == self.class_index
        # Get confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true=target, y_pred=out).ravel()
        return {f'{self.class_of_interest}.sensitivity': tp / (tp + fn),
                f'{self.class_of_interest}.specificity': tn / (tn + fp)}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_fit_start(trainer, pl_module)
        original = pl_module.get_metrics
        pl_module.get_metrics = lambda batch: {**original(batch), **self.get_metrics(pl_module, batch)}


class ClassExtraDataMetricLogger(ClassMetricLogger, ABC):
    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)
        self.dataloaders = None

    def on_fit_start(self, trainer: pl.Trainer, module: ChunkModule):
        super().on_fit_start(trainer, module)
        self.dataloaders = {name: DataLoader(dataset.for_all(), **module.cfg.dataloader)
                            for name, dataset in module.datasets.items()}

    def _on_batch_end(self, module: ChunkModule, set_name: str):
        module.log_metrics(self.get_metrics(module, next(iter(self.dataloaders[set_name]))), set_name)

    def on_train_batch_end(self, trainer: pl.Trainer, module: ChunkModule, outputs, batch, batch_idx: int,
                           unused: int = 0):
        self._on_batch_end(module, set_name='train')

    def on_validation_batch_end(self, trainer: pl.Trainer, module: ChunkModule, outputs, batch, batch_idx: int,
                                dataloader_idx: int, unused: int = 0):
        self._on_batch_end(module, set_name=module.val_dataloader_name(dataloader_idx))


class ClassMSE(ClassExtraDataMetricLogger):

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return ChunkForecaster

    def get_metrics(self, forecaster: ChunkForecaster, batch: Tuple[Tensor, Tensor, Tensor]):
        x, target, cls = batch
        out = forecaster.predict_step((torch.concat([x, target], dim=1),))
        # Only consider the specified class
        target = target[cls == self.class_index]
        out = out[cls == self.class_index, :target.size(1), :]
        return {f'{self.class_of_interest}.loss.mse': F.mse_loss(target, out)}


class ClassFeatureSTD(ClassExtraDataMetricLogger):

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return ChunkTripletFeaturizer

    def get_metrics(self, featurizer: ChunkForecaster, batch: Tuple[Tensor, Tensor, Tensor]):
        x, target, cls = batch
        out = featurizer.predict_step((x,))
        # Only consider the specified class
        out = out[cls == self.class_index]
        # Only consider the specified class
        return {f'{self.class_of_interest}.feature.std': out.std()}
