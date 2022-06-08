from ml_collections.config_dict import ConfigDict, placeholder

from config import ROOT_DIR


def get_config():
    cfg = ConfigDict()

    cfg.seed = 2022

    cfg.task = placeholder(str)

    cfg.project = placeholder(str)
    cfg.run_id = placeholder(str)

    cfg.n_in = 10
    cfg.n_out = 10
    cfg.n_features = 32

    cfg.space_dim = 3

    cfg.results_dir = f'{ROOT_DIR}/results'

    data_dir = f'{ROOT_DIR}/data/default(length=500-pts_per_period=50-resample=True-seed=2022)'
    cfg.datasets = ConfigDict({
        'train': f'{data_dir}/train(count=100-ic_noise=0.01-ic_scale=1)',
        'val': f'{data_dir}/val(count=30-ic_noise=0.01-ic_scale=1)',
        'test': f'{data_dir}/test(count=20-ic_noise=0.1-ic_scale=0.5)'
    })
    cfg.trainer = ConfigDict({
        'max_epochs': 100,
        'deterministic': True,
        'val_check_interval': 5,
        'limit_val_batches': 1,
        'log_every_n_steps': 1,
        'gpus': 1,
        'detect_anomaly': True,
    })
    cfg.early_stopping = ConfigDict({
        'patience': 5,
        'check_on_train_epoch_end': True
    })
    cfg.optimizer = ConfigDict({
        'lr': 0.01
    })
    cfg.lr_scheduler = ConfigDict({
        'patience': 2,
        'factor': 0.2
    })
    cfg.dataloader = ConfigDict({
        'batch_size': 1024,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True
    })
    cfg.models = ConfigDict({
        'MultiNBEATS': {
            'n_stacks': 4,
            'n_blocks': 4,
            'expansion_coefficient_dim': 4,
            'n_layers': 4,
            'layer_widths': 8
        },
        'MultiTransformer': {
            'd_model': 16,
            'nhead': 4,
            'dim_feedforward': 16,
            'num_encoder_layers': 4
        },
        'MultiGRU': {
            'n_hidden': 30,
            'n_layers': 2,
            'dropout': 0.1
        },
        'MultiLSTM': {
            'n_hidden': 26,
            'n_layers': 2,
            'dropout': 0.1
        }
    })
    return cfg
