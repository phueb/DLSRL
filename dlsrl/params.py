

param2default = {
    "train_batch_size": 128,
    "architecture": "interleaved",
    "use_se_marker": False,
    "embed_size": "100",
    "cell_size": 256,
    "num_layers": 8,
    "max_grad_norm": 1.0,
    "keep_prob": 0.9,
    "max_epochs": 500,
    "learning_rate": 1.0,
    "epsilon": 1e-6
}


param2requests = {
    "architecture": ['stacked', 'interleaved']
}
