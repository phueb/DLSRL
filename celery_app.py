from celery import Celery
import sys

import celeryconfig

sys.path.append('/media/lab/DLSRL')


app = Celery('dlsrl')
app.config_from_object(celeryconfig)


@app.task
def srl_task(**kwargs):
    from src.celery_task import srl_task
    return srl_task(**kwargs)