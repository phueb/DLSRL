from celery import Celery

import celeryconfig
from src.celery_task import srl_task as _srl_task

app = Celery('dlsrl')
app.config_from_object(celeryconfig)


@app.task
def srl_task(**kwargs):
    return _srl_task(**kwargs)