import unittest

import torch
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, random_split, DataLoader

from ecodyna.models.mutitask_models import MultiTaskTimeSeriesModel
from models.task_modules import ChunkClassifier
from scripts.experiments.defaults import small_models

common_parameters = dict(n_in=5, space_dim=8)


class MultiTaskTimeSeriesModelsTests(unittest.TestCase):
    def test_abstract_superclass(self):
        self.assertRaises(TypeError, lambda x: MultiTaskTimeSeriesModel(**common_parameters))

    def test_default_featurizer(self):
        for Model, params in small_models.values():
            model = Model(**params, **common_parameters)
            self.assertTrue(model.is_prepared_to_featurize())
            self.assertFalse(model.is_prepared_to_classify())
            self.assertFalse(model.is_prepared_to_forecast())
            self.assertEqual(model.n_features, model._natural_n_features)

    def test_can_build_classifier(self):
        for Model, params in small_models.values():
            model = Model(n_features=12, n_classes=10, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_classify())

    def test_can_build_featurizer(self):
        for Model, params in small_models.values():
            model = Model(n_features=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_featurize())

    def test_can_build_forecaster(self):
        for Model, params in small_models.values():
            model = Model(n_features=12, n_out=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_forecast())

    def test_can_build_all_at_once(self):
        for Model, params in small_models.values():
            model = Model(n_features=10, n_classes=10, n_out=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_classify())
            self.assertTrue(model.is_prepared_to_featurize())
            self.assertTrue(model.is_prepared_to_forecast())

    def test_nbeats_default_forecasting(self):
        Model, params = small_models['N-BEATS']
        nbeats = Model(n_out=10, **params, **common_parameters)
        self.assertTrue(nbeats.is_prepared_to_forecast())
        self.assertFalse(nbeats.is_prepared_to_featurize())

    def test_nbeats_add_forecasting(self):
        Model, params = small_models['N-BEATS']
        nbeats = Model(n_features=10, **params, **common_parameters)
        self.assertTrue(nbeats.is_prepared_to_featurize())
        self.assertFalse(nbeats.is_prepared_to_forecast())

        nbeats.prepare_to_forecast(n_out=10)
        self.assertTrue(nbeats.is_prepared_to_forecast())

    def test_classify_shape(self):
        for Model, params in small_models.values():
            model = Model(n_features=12, n_classes=10, **params, **common_parameters)
            x = torch.rand((2, model.n_in, model.space_dim))
            y = model(x, kind='classify')
            self.assertEqual(y.size(), (2, model.n_classes))

    def test_featurize_shape(self):
        for Model, params in small_models.values():
            model = Model(n_features=12, **params, **common_parameters)
            x = torch.rand((2, model.n_in, model.space_dim))
            y = model(x, kind='featurize')
            self.assertEqual(y.size(), (2, model.n_features))

    def test_forecast_shape(self):
        for Model, params in small_models.values():
            model = Model(n_features=12, n_out=10, **params, **common_parameters)
            x = torch.rand((2, model.n_in, model.space_dim))
            y = model(x, kind='forecast')
            self.assertEqual(y.size(), (2, model.n_out, model.space_dim))

    def test_prepare_to_classify(self):
        for Model, params in small_models.values():
            for task_params in [{}, {'n_features': 10}, {'n_features': 5, 'n_out': 10}]:
                model = Model(**task_params, **params, **common_parameters)
                model.prepare_to_classify(n_classes=10)
                x = torch.rand((2, model.n_in, model.space_dim))
                y = model(x, kind='classify')
                self.assertEqual(y.size(), (2, model.n_classes))

    def test_prepare_to_forecast(self):
        for Model, params in small_models.values():
            for task_params in [{}, {'n_features': 10}, {'n_features': 10, 'n_classes': 10}]:
                model = Model(**task_params, **params, **common_parameters)
                model.prepare_to_forecast(n_out=10)
                x = torch.rand((2, model.n_in, model.space_dim))
                y = model(x, kind='forecast')
                self.assertEqual(y.size(), (2, model.n_out, model.space_dim))

    def test_classifier_learns_easy(self):
        n = 100
        X = torch.rand((n, common_parameters['n_in'], common_parameters['space_dim']))
        y = torch.zeros(n).long()
        X[:n // 2, :, :] -= 1
        X[n // 2:, :, :] += 1
        y[n // 2:] = 1
        dataset = TensorDataset(X, y)
        train_dataset, val_dataset = random_split(dataset, [70, 30])
        train_dataloader = DataLoader(train_dataset, batch_size=10, num_workers=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=8)
        for Model, params in small_models.values():
            trainer = Trainer(max_epochs=200, logger=False, enable_checkpointing=False, enable_progress_bar=False)
            classifier = ChunkClassifier(model=Model(n_classes=2, **params, **common_parameters))
            trainer.fit(classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            print(trainer.callback_metrics)
            self.assertLess(0.8, trainer.callback_metrics['val_acc'].item())
        self.assertTrue(True)

    def test_classifier_doesnt_learn_hard(self):
        n = 100
        X = torch.rand((n, common_parameters['n_in'], common_parameters['space_dim']))
        y = torch.zeros(n).long()
        y[n // 3:] = 1
        y[2 * n // 3:] = 2
        dataset = TensorDataset(X, y)
        train_dataset, val_dataset = random_split(dataset, [70, 30])
        train_dataloader = DataLoader(train_dataset, batch_size=10, num_workers=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=8)
        for Model, params in small_models.values():
            trainer = Trainer(max_epochs=200, logger=False, enable_checkpointing=False, enable_progress_bar=False)
            classifier = ChunkClassifier(model=Model(n_classes=3, **params, **common_parameters))
            trainer.fit(classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            print(trainer.callback_metrics)
            self.assertLess(trainer.callback_metrics['val_acc'].item(), 0.5)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
