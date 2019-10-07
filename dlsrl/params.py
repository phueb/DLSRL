
# defaults as defined by He et al., 2017
param2default = {
    'max_sentence_length': 128,  # reduces padding and speeds training  - max is 116 (WSJ dev)
    'batch_size': 128,  # actually 90
    'num_layers': 8,
    'hidden_size': 256,  # actually, 300
    'max_grad_norm': 1.0,
    'max_epochs': 500,
    'learning_rate': 0.95,
    'epsilon': 1e-6,
    'glove': True
}

# used to overwrite parameters when --debug flag is on (when calling "ludwig-local")
param2debug = {'max_epochs': 500,
               'glove': False
               }

param2requests = {
    'max_epochs': [10, 50, 100, 200, 500]
}
