import unittest

import torch

from ecodyna.mutitask_models import MultiTaskTimeSeriesModel, MyRNN, MyNBEATS, MyTransformer

common_parameters = dict(n_in=5, space_dim=8)
models = [
    (MyRNN, {'model': 'GRU', 'n_layers': 2, 'n_hidden': 12}),
    (MyRNN, {'model': 'LSTM', 'n_layers': 2, 'n_hidden': 12}),
    (MyNBEATS, {'layer_widths': 32, 'n_layers': 4, 'expansion_coefficient_dim': 5, 'n_stacks': 2, 'n_blocks': 2}),
    (MyTransformer, {'nhead': common_parameters['space_dim']})
]


class MultiTaskTimeSeriesModelsTests(unittest.TestCase):
    def test_abstract_superclass(self):
        self.assertRaises(TypeError, lambda x: MultiTaskTimeSeriesModel(**common_parameters))

    def test_default_featurizer(self):
        for Model, params in models:
            model = Model(**params, **common_parameters)
            self.assertTrue(model.is_prepared_to_featurize())
            self.assertFalse(model.is_prepared_to_classify())
            self.assertFalse(model.is_prepared_to_forecast())

    def test_can_build_classifier(self):
        for Model, params in models:
            model = Model(n_features=12, n_classes=10, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_classify())

    def test_can_build_featurizer(self):
        for Model, params in models:
            model = Model(n_features=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_featurize())

    def test_can_build_forecaster(self):
        for Model, params in models:
            model = Model(n_features=12, n_out=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_forecast())

    def test_can_build_all_at_once(self):
        for Model, params in models:
            model = Model(n_features=10, n_classes=10, n_out=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_classify())
            self.assertTrue(model.is_prepared_to_featurize())
            self.assertTrue(model.is_prepared_to_forecast())

    def test_classify_shape(self):
        for Model, params in models:
            model = Model(n_features=12, n_classes=10, **params, **common_parameters)
            x = torch.rand((2, model.n_in, model.space_dim))
            y = model(x, kind='classify')
            self.assertTrue(y.size() == (2, model.n_classes))

    def test_featurize_shape(self):
        for Model, params in models:
            model = Model(n_features=12, **params, **common_parameters)
            x = torch.rand((2, model.n_in, model.space_dim))
            y = model(x, kind='featurize')
            self.assertTrue(y.size() == (2, model.n_features))

    def test_forecast_shape(self):
        for Model, params in models:
            model = Model(n_features=12, n_out=10, **params, **common_parameters)
            x = torch.rand((2, model.n_in, model.space_dim))
            y = model(x, kind='forecast')
            self.assertTrue(y.size() == (2, model.n_out, model.space_dim))

    def test_prepare_to_classify(self):
        for Model, params in models:
            for task_params in [{}, {'n_out': 10}]:
                model = Model(**task_params, **params, **common_parameters)
                model.prepare_to_classify(n_classes=10)
                x = torch.rand((2, model.n_in, model.space_dim))
                y = model(x, kind='classify')
                self.assertTrue(True)

    def test_prepare_to_featurize(self):
        for Model, params in models:
            for task_params in [{'n_classes': 10}, {'n_out': 10}]:
                model = Model(**task_params, **params, **common_parameters)
                model.prepare_to_featurize(n_features=model.n_features)
                x = torch.rand((2, model.n_in, model.space_dim))
                y = model(x, kind='featurize')
                self.assertTrue(True)

    def test_prepare_to_forecast(self):
        for Model, params in models:
            for task_params in [{}, {'n_classes': 10}]:
                model = Model(**task_params, **params, **common_parameters)
                model.prepare_to_forecast(n_out=10)
                x = torch.rand((2, model.n_in, model.space_dim))
                y = model(x, kind='forecast')
                self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
