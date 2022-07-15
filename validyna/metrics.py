from abc import ABC, abstractmethod
from typing import Tuple, Type, Any

import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import Tensor

import validyna.registry as registry
from validyna.models.task_modules import ChunkClassifier, ChunkModule, ChunkForecaster, ChunkTripletFeaturizer


def with_prober_class(f):
    def wrapper(self, *args, **kwargs):
        original_class = self.module.__class__
        self.module.__class__ = self.cls
        value = f(self, *args, **kwargs)
        self.module.__class__ = original_class
        return value
    return wrapper


class Prober(pl.Callback):
    def __init__(self, task: str):
        self.task = task
        self.cls: Type[ChunkModule] = registry.task_registry[task]
        self.module = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if isinstance(pl_module, self.cls):
            raise ValueError("The prober should be for a task different than the training one.")
        self.module = pl_module

    @with_prober_class
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, unused: int = 0):
        self.module.training_step(batch, batch_idx)

    @with_prober_class
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int):
        self.module.validation_step(batch, batch_idx, dataloader_idx)


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
        self.class_index = pl_module.datasets['train'].classes.index(self.class_of_interest)

        original = pl_module.get_metrics
        pl_module.get_metrics = lambda batch, set_name: {**original(batch, set_name),
                                                         **self.get_metrics(pl_module, batch)}


class ClassSensitivitySpecificity(ClassMetricLogger):

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return ChunkClassifier

    def get_metrics(self, classifier: ChunkClassifier, batch: Tuple[Tensor, Tensor, Tensor]):
        x_in, x_out, x_class = batch
        pred = classifier.predict_step((x_in,))
        # Ignore all different classes and only consider this class or any other
        target = x_class == self.class_index
        pred = pred == self.class_index
        # Get confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true=target.to('cpu'), y_pred=pred.to('cpu')).ravel()
        return {f'{self.class_of_interest}.sensitivity': tp / (tp + fn),
                f'{self.class_of_interest}.specificity': tn / (tn + fp)}


class ClassMSE(ClassMetricLogger):

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return ChunkForecaster

    def get_metrics(self, forecaster: ChunkForecaster, batch: Tuple[Tensor, Tensor, Tensor]):
        x_in, x_out, x_class = batch
        pred = forecaster(x_in)
        # Only consider the specified class
        target = x_out[x_class == self.class_index]
        pred = pred[x_class == self.class_index, :target.size(1), :]
        return {f'{self.class_of_interest}.loss.mse': F.mse_loss(target, pred)}


class ClassFeatureSTD(ClassMetricLogger):

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return ChunkTripletFeaturizer

    def get_metrics(self, featurizer: ChunkForecaster, batch: Tuple[Tensor, Tensor, Tensor]):
        x_in, x_out, x_class = batch
        pred = featurizer.predict_step((x_in,))
        # Only consider the specified class
        pred = pred[x_class == self.class_index]
        # Only consider the specified class
        return {f'{self.class_of_interest}.feature.std': pred.std()}
