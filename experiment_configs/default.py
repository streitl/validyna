from ml_collections.config_dict import ConfigDict, placeholder

from config import ROOT_DIR


def get_config():
    cfg = ConfigDict()

    cfg.seed = 2022
    cfg.gpus = 1

    cfg.task = placeholder(str, required=True)
    cfg.project = placeholder(str, required=True)

    cfg.n_in = 10
    cfg.n_out = 10
    cfg.n_features = 32

    cfg.run_id = None
    cfg.space_dim = 3

    cfg.results_dir = f'{ROOT_DIR}/results'

    cfg.datasets = ConfigDict({
        'train': f'{ROOT_DIR}/data/default/train(count=100-ic_noise=0.01-ic_scale=1)',
        'val': f'{ROOT_DIR}/data/default/val(count=30-ic_noise=0.01-ic_scale=1)',
        'test': f'{ROOT_DIR}/data/default/test(count=20-ic_noise=10-ic_scale=0.5)'
    })
    cfg.trainer = ConfigDict({
        'max_epochs': 100,
        'deterministic': True,
        'val_check_interval': 5,
        'limit_val_batches': 1,
        'log_every_n_steps': 1,
        'gpus': 0
    })
    cfg.early_stopping = ConfigDict({
        'patience': 5
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
        'num_workers': 8,
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
