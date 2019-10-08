import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import sys
import numpy as np
import pandas as pd

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator


from dlsrl.data import Data
from dlsrl.eval import evaluate_model_on_dev
from dlsrl.models import make_model_and_optimizer
from dlsrl import config


class Params:

    def __init__(self, param2val):
        param2val = param2val.copy()

        self.param_name = param2val.pop('param_name')
        self.job_name = param2val.pop('job_name')

        self.param2val = param2val

    def __getattr__(self, name):
        if name in self.param2val:
            return self.param2val[name]
        else:
            raise AttributeError('No such attribute')

    def __str__(self):
        res = '\nParams:'
        for k, v in sorted(self.param2val.items()):
            res += '{}={}\n'.format(k, v)
        return res


def main(param2val):

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # make local folder for saving checkpoint + events files
    local_job_p = config.LocalDirs.runs / params.job_name
    if not local_job_p.exists():
        local_job_p.mkdir(parents=True)

    # if not run on Ludwig
    if config.Global.local:
        print('WARNING: Loading data locally because config.Global.local=True')
        config.RemoteDirs = config.LocalDirs

    # prefer loading glove embeddings from local machine, if they are present
    if config.LocalDirs.glove.exists():
        glove_path = config.LocalDirs.glove
    else:
        glove_path = config.RemoteDirs.glove

    # data + vocab + batcher
    data = Data(params)
    vocab = Vocabulary.from_instances(data.train_instances + data.dev_instances)
    bucket_batcher = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(vocab)

    # model + optimizer
    model, optimizer = make_model_and_optimizer(params, vocab, glove_path)

    # train + eval loop
    dev_f1s = []
    train_start = time.time()
    for epoch in range(params.max_epochs):

        print('\nEpoch: {}'.format(epoch))

        # eval on dev propositions
        dev_f1 = evaluate_model_on_dev(model, params, data, vocab, bucket_batcher)
        dev_f1s.append(dev_f1)

        # train
        model.train()
        train_generator = bucket_batcher(data.train_instances, num_epochs=1)
        for step, batch in enumerate(train_generator):

            batch['training'] = True
            loss = model.train_on_batch(batch, optimizer)

            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    # to pandas
    eval_epochs = np.arange(params.max_epochs)
    df_dev_f1 = pd.Series(dev_f1s, index=eval_epochs)
    df_dev_f1.name = 'dev_f1'

    return [df_dev_f1]
