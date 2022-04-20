from ecodyna.experiment_helper import run_forecasting_experiment
from ecodyna.metrics import RNNForecastMetricLogger
from ecodyna.mutitask_models import MultiTaskRNN

if __name__ == '__main__':
    # parameters
    data_parameters = {'trajectory_count': 1000, 'trajectory_length': 100}
    in_out_parameters = {'n_in': 5, 'n_out': 5}
    common_model_parameters = {'model': 'LSTM', 'n_hidden': 32, 'n_layers': 1, **in_out_parameters}
    experiment_parameters = {'n_epochs': 5, 'train_part': 0.75, 'n_splits': 2}
    dataloader_parameters = {'batch_size': 64, 'num_workers': 8, **in_out_parameters}

    models_and_params = [
        (MultiTaskRNN, {'forecast_type': 'n_out'}),
        (MultiTaskRNN, {'forecast_type': 'recurrent'})
    ]

    run_forecasting_experiment(
        project='lstm-forecasting-comparison-1',
        data_parameters=data_parameters,
        common_model_parameters=common_model_parameters,
        experiment_parameters=experiment_parameters,
        dataloader_parameters=dataloader_parameters,
        models_and_params=models_and_params,
        trainer_callbacks=[RNNForecastMetricLogger]
    )
