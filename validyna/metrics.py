import pdb
from abc import ABC, abstractmethod
from typing import Tuple, Type, Any

import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import Tensor

import validyna.registry as registry
from validyna.models.task_modules import SliceClassifier, SliceModule, SliceForecaster, SliceFeaturizer


def with_prober_class(f):
    def wrapper(self, *args, **kwargs):
        original_class = self.module.__class__
        self.module.__class__ = self.cls
        value = f(self, *args, **kwargs)
        self.module.__class__ = original_class
        return value

    return wrapper


class Prober(pl.Callback):
    """
    Probes the metrics corresponding to a task different to the one the model is being trained for.
    For instance, while training for classification, can proble feature extraction metrics such as triplet loss.
    """

    def __init__(self, task: str):
        self.task = task
        self.cls: Type[SliceModule] = registry.task_registry[task]
        self.module = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if isinstance(pl_module, self.cls):
            raise ValueError('The prober should be for a task different than the training one.')
        self.module = pl_module

    @with_prober_class
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, unused: int = 0):
        self.module.training_step(batch, batch_idx)

    @with_prober_class
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int):
        self.module.validation_step(batch, batch_idx, dataloader_idx)


class ClassMetricLogger(pl.Callback, ABC):
    """
    Super-class of metric loggers that are specific to one class and that extend the module's <get_metrics> method.
    """

    def __init__(self, class_of_interest: str):
        self.class_of_interest = class_of_interest
        self.class_index = None

    @abstractmethod
    def required_module_type(self) -> Type[SliceModule]:
        pass

    @abstractmethod
    def get_metrics(self, pl_module: pl.LightningModule, batch: Any) -> dict[str, float]:
        pass

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(pl_module, SliceModule):
            raise ValueError(f'This callback only works for {SliceModule.__name__}')
        if not isinstance(pl_module, self.required_module_type()):
            raise ValueError(
                f'Can only log the metric {self.__class__.__name__} for class {self.required_module_type().__name__}')
        self.class_index = pl_module.datasets['train'].classes.index(self.class_of_interest)

        original = pl_module.get_metrics
        pl_module.get_metrics = lambda batch, set_name: {**original(batch, set_name),
                                                         **self.get_metrics(pl_module, batch)}


class ClassSensitivitySpecificity(ClassMetricLogger):
    """
    Logs the sensitivity (a.k.a. true positive rate) and specificity (true negative rate) of the specified class.
    """

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return SliceClassifier

    def get_metrics(self, classifier: SliceClassifier, batch: Tuple[Tensor, Tensor, Tensor]):
        x_in, x_out, x_class = batch
        pred: Tensor = classifier.predict_step((x_in,))
        # Ignore all different classes and only consider this class vs any other
        target = x_class == self.class_index
        pred = pred == self.class_index
        # Get confusion matrix metrics
        res = confusion_matrix(y_true=target.to('cpu'), y_pred=pred.to('cpu')).ravel()
        sensitivity, specificity = 1.0, 1.0
        if len(res) == 4:
            tn, fp, fn, tp = res
            if tp + fn != 0:
                sensitivity = tp / (tp + fn)
            if tn + fp != 0:
                specificity = tn / (tn + fp)
        # pdb.set_trace()
        return {f'{self.class_of_interest}.sensitivity': sensitivity,
                f'{self.class_of_interest}.specificity': specificity}


class ClassMSE(ClassMetricLogger):
    """
    Logs the Mean Squared Error of the model but only for trajectories of the specified class.
    """

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return SliceForecaster

    def get_metrics(self, forecaster: SliceForecaster, batch: Tuple[Tensor, Tensor, Tensor]):
        x_in, x_out, x_class = batch
        pred = forecaster(x_in)
        # Only consider the specified class
        target = x_out[x_class == self.class_index]
        pred = pred[x_class == self.class_index, :target.size(1), :]
        return {f'{self.class_of_interest}.loss.mse': F.mse_loss(target, pred)}


class ClassFeatureSTD(ClassMetricLogger):
    """
    Logs the standard deviation of features of the model but only for samples of the specified class.
    """

    def __init__(self, class_of_interest: str):
        super().__init__(class_of_interest)

    def required_module_type(self):
        return SliceFeaturizer

    def get_metrics(self, featurizer: SliceForecaster, batch: Tuple[Tensor, Tensor, Tensor]):
        x_in, x_out, x_class = batch
        pred = featurizer.predict_step((x_in,))
        # Only consider the specified class
        pred = pred[x_class == self.class_index]
        # Only consider the specified class
        return {f'{self.class_of_interest}.feature.std': pred.std()}
