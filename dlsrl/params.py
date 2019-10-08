
param2default = {
    # defaults as defined by He et al., 2017
    'batch_size': 128,  # actually 90
    'num_layers': 8,
    'hidden_size': 256,  # actually, 300
    'max_grad_norm': 1.0,
    'max_epochs': 500,
    'learning_rate': 0.95,
    'epsilon': 1e-6,
    'binary_feature_dim': 100,
    'max_sentence_length': 128,  # needs to be larger than max=116 (WSJ dev)
    'glove': True,
    'my_implementation': False
}

# used to overwrite parameters when --debug flag is on (when calling "ludwig-local")
param2debug = {'glove': False}

param2requests = {
    'my_implementation': [False, True],
    'max_epochs': [50]
}

if 'nun_layers' in param2requests:
    for num_layers in param2requests['num_layers']:
        assert num_layers % 2 == 0  # because of bi-directional organization
        assert num_layers >= 2
