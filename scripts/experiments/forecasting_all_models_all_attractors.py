from pytorch_lightning.callbacks import EarlyStopping

from ecodyna.experiment_helper import run_forecasting_experiment
from ecodyna.metrics import ForecastMetricLogger
from ecodyna.mutitask_models import MultiTaskRNN, MultiTaskNBEATS

if __name__ == '__main__':
    # parameters
    data_parameters = {'trajectory_count': 1000, 'trajectory_length': 100}
    in_out_parameters = {'n_in': 5, 'n_out': 5}
    common_model_parameters = {**in_out_parameters}
    experiment_parameters = {'max_epochs': 2, 'train_part': 0.75, 'n_splits': 2}
    trainer_parameters = {'max_epochs': 2, 'callbacks': [EarlyStopping('val_loss', patience=5)]}
    dataloader_parameters = {'batch_size': 64, 'num_workers': 8}

    models_and_params = [
        (MultiTaskRNN, {'model': 'LSTM', 'n_hidden': 32, 'n_layers': 1}),
        (MultiTaskRNN, {'model': 'GRU', 'n_hidden': 32, 'n_layers': 1}),
        (MultiTaskNBEATS, {'n_stacks': 4, 'n_blocks': 4})
    ]

    run_forecasting_experiment(
        project='chaos-forecasting',
        data_parameters=data_parameters,
        in_out_parameters=in_out_parameters,
        common_model_parameters=common_model_parameters,
        experiment_parameters=experiment_parameters,
        trainer_parameters=trainer_parameters,
        dataloader_parameters=dataloader_parameters,
        models_and_params=models_and_params,
        metric_loggers=[ForecastMetricLogger]
    )
