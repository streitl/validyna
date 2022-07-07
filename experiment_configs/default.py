from typing import Optional, Union

from ml_collections.config_dict import ConfigDict, placeholder

from config import ROOT_DIR
from validyna.data import make_datasets


def get_config(which: Optional[str] = None) -> Union[ConfigDict, dict]:

    placeholders = {
        'project': placeholder(str),
        'tasks-common-task': placeholder(str),
        'tasks-common-run': placeholder(str),
    }

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

    data_dir = f'{ROOT_DIR}/data/default(length=200-pts_per_period=50-resample=True-seed=2022)'
    cfg.tasks = ConfigDict({
        'list': [],
        'common': ConfigDict({
            'datasets': lambda: make_datasets({
                'train': f'{data_dir}/train(count=100-ic_noise=0.01-ic_scale=1)',
                'val': f'{data_dir}/val(count=20-ic_noise=0.01-ic_scale=1)',
                'test': f'{data_dir}/test(count=30-ic_noise=0.05-ic_scale=1.001)',
            }, cfg.n_in, cfg.n_out),
            'task': placeholders['tasks-common-task'],
            'run': placeholders['tasks-common-run'],
        })
    })
    cfg.trainer = ConfigDict({
        'max_epochs': 100,
        'deterministic': True,
        'val_check_interval': 5,
        'limit_val_batches': 1.0,
        'limit_train_batches': 1.0,
        'log_every_n_steps': 1,
        'gpus': 1,
        'detect_anomaly': True,
        'fast_dev_run': False,
    }, type_safe=False)
    cfg.early_stopping = ConfigDict({'patience': 3, 'check_on_train_epoch_end': True})
    cfg.optimizer = ('AdamW', {'lr': 0.01})
    cfg.lr_scheduler = ('ReduceLROnPlateau', {'patience': 1, 'factor': 0.2})
    cfg.normalize_data = False
    cfg.dataloader = ConfigDict({
        'batch_size': 1024,
        'num_workers': 4,
        'shuffle': True,
        'persistent_workers': True,
        'pin_memory': True,
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
        })
    ]
    return cfg
