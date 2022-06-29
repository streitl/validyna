from typing import Type

import pytorch_lightning as pl

from validyna.models.multitask_models import all_implementations, MultiTaskTimeSeriesModel
from validyna.models.task_modules import ChunkClassifier, ChunkTripletFeaturizer, ChunkForecaster

task_registry: dict[str, Type[pl.LightningModule]] = dict()
model_registry: dict[str, Type[MultiTaskTimeSeriesModel]] = dict()
metrics_registry: dict[str, ...] = dict()  # TODO


def register_model(name: str, Model: Type[MultiTaskTimeSeriesModel]):
    assert name not in model_registry, f'{name} is already registered as a model!'
    model_registry[name] = Model


def register_task(task: str, Module: Type[pl.LightningModule]):
    assert task not in task_registry, f'{task} is already registered as a task!'
    task_registry[task] = Module


def register_metric(name: str, metric: any):
    assert name not in metrics_registry, f'{name} is already registered as a metric!'
    metrics_registry[name] = metric


for Model in all_implementations:
    register_model(Model.name(), Model)

register_task('classification', ChunkClassifier)
register_task('featurization', ChunkTripletFeaturizer)
register_task('forecasting', ChunkForecaster)
