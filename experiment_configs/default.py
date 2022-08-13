from typing import Optional, Union

from ml_collections.config_dict import ConfigDict, placeholder

from config import ROOT_DIR
from validyna.data import load_datasets

placeholders = {
    'project': placeholder(str),
    'task': placeholder(str),
    'run_suffix': placeholder(str),
}


def get_config(which: Optional[str] = None) -> Union[ConfigDict, dict]:

    if which == 'placeholders':
        return placeholders

    cfg = ConfigDict()

    cfg.seed = 2022
    cfg.project = placeholders['project']

    cfg.n_in = 5
    cfg.n_out = 5
    cfg.n_features = 32

    cfg.space_dim = 3

    cfg.use_wandb = True
    cfg.results_dir = f'{ROOT_DIR}/results'

    cfg.scale_data = True

    data_dir = f'{ROOT_DIR}/data/default(length=250-pts_per_period=50-resample=True)'
    cfg.datasets = lambda: load_datasets({
        'train': f'{data_dir}/train(count=80-ic_noise=0.05-seed=0)',
        'val': f'{data_dir}/val(count=20-ic_noise=0.05-seed=1)',
        'test': f'{data_dir}/test(count=30-ic_noise=0.1-seed=2)',
    })
    cfg.task = placeholders['task']
    cfg.run_suffix = placeholders['run_suffix']
    cfg.runs = []

    cfg.trainer = ConfigDict({
        'max_epochs': 100,
        'deterministic': True,
        'val_check_interval': 1,
        'limit_val_batches': 1,
        'limit_train_batches': 1.0,
        'log_every_n_steps': 1,
        'gpus': 1,
        'detect_anomaly': True,
        'track_grad_norm': 2,
        'fast_dev_run': False,
        'callbacks': [],
    }, type_safe=False)
    cfg.early_stopping = ConfigDict({'patience': 3, 'check_on_train_epoch_end': True})
    cfg.optimizer = ('AdamW', {'lr': 0.01})
    cfg.lr_scheduler = ('ReduceLROnPlateau', {'patience': 1, 'factor': 0.2})
    cfg.dataloader = ConfigDict({
        'batch_size': 1024,
        'num_workers': 4,
        'shuffle': True,
        'persistent_workers': True,
    }, type_safe=False)
    cfg.models = [
        ('N-BEATS', {
            'n_stacks': 4,
            'n_blocks': 4,
            'expansion_coefficient_dim': 4,
            'n_layers': 4,
            'layer_widths': 8,
        }),
        ('Transformer', {
            'd_model': 16,
            'nhead': 4,
            'dim_feedforward': 16,
            'num_encoder_layers': 4,
        }),
        ('GRU', {
            'n_hidden': 30,
            'n_layers': 2,
            'dropout': 0.1,
        }),
        ('LSTM', {
            'n_hidden': 26,
            'n_layers': 2,
            'dropout': 0.1,
        }),
    ]
    return cfg
