from ml_collections.config_dict import ConfigDict, placeholder
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ROOT_DIR
from validyna.models.multitask_models import MultiNBEATS, MultiTransformer, MultiGRU, MultiLSTM


def get_config():
    cfg = ConfigDict()

    cfg.seed = 2022

    cfg.project = placeholder(str)

    cfg.n_in = 50
    cfg.n_out = 50
    cfg.n_features = 32

    cfg.space_dim = 3

    cfg.use_wandb = True
    cfg.results_dir = f'{ROOT_DIR}/results'

    data_dir = f'{ROOT_DIR}/data/default(length=1000-pts_per_period=100-resample=True-seed=2022)'
    cfg.tasks = ConfigDict({
        'list': [],
        'common': ConfigDict({
            'datasets': {
                'train': f'{data_dir}/train(count=100-ic_noise=0.01-ic_scale=1)',
                'val': f'{data_dir}/val(count=20-ic_noise=0.01-ic_scale=1)',
                'test': f'{data_dir}/test(count=30-ic_noise=0.05-ic_scale=1.001)',
            }
        })
    })
    cfg.trainer = ConfigDict({
        'max_epochs': 100,
        'deterministic': True,
        'val_check_interval': 5,
        'limit_val_batches': 1,
        'log_every_n_steps': 1,
        'gpus': 1,
        'detect_anomaly': True,
        'fast_dev_run': True,
    })
    cfg.early_stopping = ConfigDict({'patience': 3, 'check_on_train_epoch_end': True})
    cfg.optimizer = (AdamW, {'lr': 0.01})
    cfg.lr_scheduler = (ReduceLROnPlateau, {'patience': 1, 'factor': 0.2})
    cfg.normalize_data = True
    cfg.dataloader = ConfigDict({
        'batch_size': 1024,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True,
    })
    cfg.models = [
        (MultiNBEATS, {
            'n_stacks': 4,
            'n_blocks': 4,
            'expansion_coefficient_dim': 4,
            'n_layers': 4,
            'layer_widths': 8,
        }),
        (MultiTransformer, {
            'd_model': 16,
            'nhead': 4,
            'dim_feedforward': 16,
            'num_encoder_layers': 4,
        }),
        (MultiGRU, {
            'n_hidden': 30,
            'n_layers': 2,
            'dropout': 0.1,
        }),
        (MultiLSTM, {
            'n_hidden': 26,
            'n_layers': 2,
            'dropout': 0.1,
        })
    ]
    return cfg
