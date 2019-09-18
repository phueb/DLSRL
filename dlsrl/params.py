

param2default = {
    "batch_size": 128,  # He et al., 2017 (actually, 90)
    "max_sentence_length": 100,  # reduces padding and speeds training
    "cell_size": 256,  # He et al., 2017 (actually, 300)
    "max_grad_norm": 1.0,  # He et al., 2017
    "dropout_prob": 0.1,  # He et al., 2017
    "max_epochs": 500,  # He et al., 2017
    "learning_rate": 0.95,  # He et al., 2017
    "epsilon": 1e-6  # He et al., 2017
}


param2requests = {
    "cell_size": [256, 256, 256, 256]
}
