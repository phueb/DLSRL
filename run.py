import argparse

from src.celery_task import srl_task

# overwrite configs
parser = argparse.ArgumentParser()
parser.add_argument('-a', action="store", dest='architecture', type=str, required=True,
                    help='Model architecture')
parser.add_argument('-c', action="store", dest='cell_size', type=int, required=True,
                    help='LSTM cell size')
namespace = parser.parse_args()


# srl
srl_task(**vars(namespace))