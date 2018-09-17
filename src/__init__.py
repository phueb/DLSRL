import socket
from dotenv import load_dotenv
from pathlib import Path


# load environment variables
p = Path(__file__).parent.parent / '.env'
if p.exists():
    load_dotenv(dotenv_path=str(p), verbose=True, override=True)
    print('Loaded environment variables from {}'.format(p))
else:
    raise SystemExit('Did not find environment file')


config_str = '''
{
  "train_batch_size" : 128,
  "architecture" : "n/a", 
  "use_se_marker": false,
  "embed_size" : "100",
  "cell_size": 256,
  "num_layers" : 8,
  "max_grad_norm": 1.0,
  "keep_prob" : 0.9,
  "max_epochs": 500,
  "learning_rate": 1.0,
  "epsilon": 1e-6
}
'''