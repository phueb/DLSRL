

param2default = {
    "batch_size": 128,
    "use_se_marker": False,  # sentence start and end symbols
    "embed_size": "100",
    "cell_size": 256,
    "num_layers": 2,  # TODO change back to 8
    "max_grad_norm": 1.0,  # He et al., 2017
    "keep_prob": 0.9,  # He et al., 2017
    "max_epochs": 500,  # He et al., 2017
    "learning_rate": 0.95,  # He et al., 2017
    "epsilon": 1e-6  # He et al., 2017
}


param2requests = {
    "cell_size": [256]
}
