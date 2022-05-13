from ecodyna.models.mutitask_models import MyNBEATS, MyTransformer, MyRNN

all_models = [
    (MyNBEATS, {
        'n_stacks': 4, 'n_blocks': 2, 'expansion_coefficient_dim': 5, 'n_layers': 4, 'layer_widths': 16
    }),
    (MyTransformer, {'d_model': 16, 'nhead': 4, 'dim_feedforward': 32, 'num_encoder_layers': 4}),
    (MyRNN, {'model': 'LSTM', 'n_hidden': 64, 'n_layers': 1, 'dropout': 0.1}),
    (MyRNN, {'model': 'GRU', 'n_hidden': 64, 'n_layers': 1, 'dropout': 0.1}),
]
