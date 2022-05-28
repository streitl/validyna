from models.defaults import params
from itertools import product
from tqdm import tqdm
from ecodyna.models import mutitask_models

for size in ['small', 'medium', 'large']:
    print(size)
    results = {}
    for n_in, space_dim, n_classes, n_features, n_out in \
            tqdm(list(product(
                range(5, 20, 5), range(3, 12, 3), range(5, 20, 5), range(5, 20, 5), range(5, 20, 5)
            ))):
        for model_name, model_parameters in params.items():
            Model = getattr(mutitask_models, model_name)
            model = Model(n_in=n_in, space_dim=space_dim,
                          n_out=n_out, n_features=n_features, n_classes=n_classes,
                          **{k: p[size] for k, p in model_parameters.items()})
            n_params = sum(param.numel() for param in model.parameters())
            if hasattr(model, 'forecaster'): del model.forecaster
            if hasattr(model, 'classifier'): del model.classifier
            if hasattr(model, 'featurizer'): del model.featurizer
            results[model.name()] = (results[model.name()] if model.name() in results else []) + [n_params]
    print(size, {k: sum(v) / len(v) for k, v in results.items()})
