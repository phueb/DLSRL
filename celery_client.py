from pathlib import Path
import json

from celery_app import srl_task
from src import config_str


if __name__ == "__main__":
    for n in range(4):
        srl_task.delay(architecture='interleaved')

    for n in range(4, 8):
        srl_task.delay(architecture='stacked')