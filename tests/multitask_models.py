import unittest

import torch

from ecodyna.mutitask_models import MultiTaskTimeSeriesModel, MultiTaskRNN, MultiTaskNBEATS, MultiTaskTransformer

models = [
    (MultiTaskRNN, {'model': 'GRU', 'n_layers': 2, 'n_hidden': 12}),
    (MultiTaskRNN, {'model': 'LSTM', 'n_layers': 2, 'n_hidden': 12}),
    (MultiTaskNBEATS, {'expansion_coefficient_dim': 5, 'n_stacks': 2, 'n_blocks': 2}),
    (MultiTaskTransformer, {})
]
common_parameters = dict(n_in=5, space_dim=5)


class MyTestCase(unittest.TestCase):
    def test_abstract_superclass(self):
        self.assertRaises(TypeError, lambda x: MultiTaskTimeSeriesModel(**common_parameters))

    def test_can_build_classifier(self):
        for Model, params in models:
            model = Model(n_classes=10, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_classify())

    def test_can_build_featurizer(self):
        for Model, params in models:
            model = Model(n_features=12, **common_parameters)
            self.assertTrue(model.is_prepared_to_featurize())

    def test_can_build_forecaster(self):
        for Model, params in models:
            model = Model(n_out=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_forecast())

    def test_can_build_all_at_once(self):
        for Model, params in models:
            model = Model(n_classes=10, n_features=10, n_out=12, **params, **common_parameters)
            self.assertTrue(model.is_prepared_to_classify())
            self.assertTrue(model.is_prepared_to_featurize())
            self.assertTrue(model.is_prepared_to_forecast())

    def test_classify_shape(self):
        for Model, params in models:
            model = Model(n_classes=10, **params, **common_parameters)
            x = torch.rand((2, 4, model.space_dim))
            y = model(x, kind='classify')
            self.assertTrue(y.size() == (2, 4, model.n_classes))

    def test_featurize_shape(self):
        for Model, params in models:
            model = Model(n_features=10, **params, **common_parameters)
            x = torch.rand((2, 4, model.space_dim))
            y = model(x, kind='featurize')
            self.assertTrue(y.size() == (2, 4, model.n_classes))

    def test_forecast_shape(self):
        for Model, params in models:
            model = Model(n_out=10, **params, **common_parameters)
            x = torch.rand((2, 4, model.space_dim))
            y = model(x, kind='forecast')
            self.assertTrue(y.size() == (2, model.n_out, model.space_dim))


if __name__ == '__main__':
    unittest.main()
