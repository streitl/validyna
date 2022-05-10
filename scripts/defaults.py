from ecodyna.mutitask_models import MyNBEATS, MyTransformer, MyRNN

all_models = [
    (MyNBEATS, {
        'n_stacks': 4, 'n_blocks': 2, 'expansion_coefficient_dim': 5, 'n_layers': 4, 'layer_widths': 16
    }),
    (MyTransformer, {}),
    (MyRNN, {'model': 'LSTM', 'n_hidden': 32, 'n_layers': 1}),
    (MyRNN, {'model': 'GRU', 'n_hidden': 32, 'n_layers': 1}),
]
