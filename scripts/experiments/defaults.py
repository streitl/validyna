import ecodyna.models.mutitask_models as mm

params = {
    'MyNBEATS': {
        'n_stacks': {
            'small': 4,
            'medium': 8,
            'large': 12
        },
        'n_blocks': {
            'small': 2,
            'medium': 4,
            'large': 8
        },
        'expansion_coefficient_dim': {
            'small': 4,
            'medium': 6,
            'large': 8
        },
        'n_layers': {
            'small': 2,
            'medium': 4,
            'large': 8
        },
        'layer_widths': {
            'small': 8,
            'medium': 18,
            'large': 32
        }
    },
    'MyTransformer': {
        'd_model': {
            'small': 16,
            'medium': 40,
            'large': 128
        },
        'nhead': {
            'small': 4,
            'medium': 20,
            'large': 32
        },
        'dim_feedforward': {
            'small': 16,
            'medium': 64,
            'large': 64
        },
        'num_encoder_layers': {
            'small': 4,
            'medium': 8,
            'large': 12
        }
    },
    'MyGRU': {
        'n_hidden': {
            'small': 30,
            'medium': 69,
            'large': 150
        },
        'n_layers': {
            'small': 2,
            'medium': 4,
            'large': 8
        },
        'dropout': {
            'small': 0.1,
            'medium': 0.1,
            'large': 0.1
        }
    },
    'MyLSTM': {
        'n_hidden': {
            'small': 26,
            'medium': 58,
            'large': 128
        },
        'n_layers': {
            'small': 2,
            'medium': 4,
            'large': 8
        },
        'dropout': {
            'small': 0.1,
            'medium': 0.1,
            'large': 0.1
        }
    }
}

small_models = [(getattr(mm, name), {k: v['small'] for k, v in p.items()}) for name, p in params.items()]
medium_models = [(getattr(mm, name), {k: v['medium'] for k, v in p.items()}) for name, p in params.items()]
large_models = [(getattr(mm, name), {k: v['large'] for k, v in p.items()}) for name, p in params.items()]
