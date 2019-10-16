
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

    # params not relevant to He et al, 2017
    'max_sentence_length': 128,  # reduces padding and speeds training  - max is 116 (WSJ dev)
    'glove': True,
    'model': 2  # 1 is my implementation, 2 is He et al., 2017, 3 is new Bert-based SRL model
}

# used to overwrite parameters when --debug flag is on (when calling "ludwig-local")
param2debug = {'max_epochs': 500,
               'glove': False
               }

param2requests = {
    'model': [3],
    'max_epochs': [50]
}

# TODO move BERT-model to separate project (after generating a figure comparing model 2 vs 3)

if 'nun_layers' in param2requests:
    for num_layers in param2requests['num_layers']:
        assert num_layers % 2 == 0  # because of bi-directional organization
        assert num_layers >= 2


for model in param2requests['model']:
    if model not in [1, 2, 3]:
        raise AttributeError('Invalid arg to model. Must be in [1, 2, 3]')