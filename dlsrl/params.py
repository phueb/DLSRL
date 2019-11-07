
param2default = {
    # defaults as defined by He et al., 2017
    'batch_size': 128,  # actually 90
    'num_layers': 8,
    'hidden_size': 256,  # actually, 300
    'max_grad_norm': 1.0,
    'num_epochs': 500,
    'learning_rate': 0.95,
    'epsilon': 1e-6,
    'binary_feature_dim': 100,

    # params not relevant to He et al, 2017
    'max_sentence_length': 128,  # reduces padding and speeds training  - max is 116 (WSJ dev)
    'glove': True,
    'model': 2  # 1 is my implementation, 2 is He et al., 2017
}

# used to overwrite parameters when --debug flag is on (when calling "ludwig-local")
param2debug = {'num_epochs': 2,
               'glove': False,
               'num_layers': 2,
               }

param2requests = {
    'model': [2],
    'num_epochs': [50]
}

if 'num_layers' in param2requests:
    for num_layers in param2requests['num_layers']:
        assert num_layers % 2 == 0  # because of bi-directional organization
        assert num_layers >= 2


for model in param2requests['model']:
    if model not in [1, 2]:
        raise AttributeError('Invalid arg to model. Must be in [1, 2]')