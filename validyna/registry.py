from typing import Type

import pytorch_lightning as pl

from validyna.metrics import ClassSensitivitySpecificity, ClassFeatureSTD, ClassMSE, Prober
from validyna.models.multitask_models import all_implementations, MultiTaskTimeSeriesModel
from validyna.models.task_modules import SliceClassifier, SliceFeaturizer, SliceForecaster, SliceModule, SliceHybrid

task_registry: dict[str, Type[SliceModule]] = dict()
model_registry: dict[str, Type[MultiTaskTimeSeriesModel]] = dict()
metric_registry: dict[str, Type[pl.Callback]] = dict()


def register_model(name: str, Model: Type[MultiTaskTimeSeriesModel]):
    assert name not in model_registry, f'{name} is already registered as a model!'
    model_registry[name] = Model


def register_task(task: str, Module: Type[SliceModule]):
    assert task not in task_registry, f'{task} is already registered as a task!'
    task_registry[task] = Module


def register_metric(name: str, metric: any):
    assert name not in metric_registry, f'{name} is already registered as a metric!'
    metric_registry[name] = metric


for Model in all_implementations:
    register_model(Model.name(), Model)

register_task('classification', SliceClassifier)
register_task('featurization', SliceFeaturizer)
register_task('forecasting', SliceForecaster)
register_task('all', SliceHybrid)

register_metric('ClassSensitivitySpecificity', ClassSensitivitySpecificity)
register_metric('ClassFeatureSTD', ClassFeatureSTD)
register_metric('ClassMSE', ClassMSE)
register_metric('Prober', Prober)
