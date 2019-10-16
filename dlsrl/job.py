import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import sys
import numpy as np
import pandas as pd

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator


from dlsrl.data import Data
from dlsrl.eval import evaluate_model_on_f1
from dlsrl.models import make_model_and_optimizer
from dlsrl import config


class Params:

    def __init__(self, param2val):
        param2val = param2val.copy()

        if param2val['model'] == 3:
            param2val = {
                'param_name': param2val['param_name'],
                'job_name': param2val['job_name'],
                'model': 3,
                'max_epochs': 15,  # BERT-based model needs only 15
                'batch_size': 32,   # BERT-based model needs 32
                'max_sentence_length': 48}  # otherwise error

        self.param_name = param2val.pop('param_name')
        self.job_name = param2val.pop('job_name')

        self.param2val = param2val

    def __getattr__(self, name):
        if name in self.param2val:
            return self.param2val[name]
        else:
            raise AttributeError('No such attribute')

    def __str__(self):
        res = '\nParams:\n'
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
    print('Vocab size={:,}'.format(vocab.get_vocab_size('tokens')))

    # model + optimizer
    model, optimizer = make_model_and_optimizer(params, vocab, glove_path)

    # train + eval loop
    dev_f1s = []
    train_f1s = []
    train_start = time.time()
    for epoch in range(params.max_epochs):

        print('\nEpoch: {}'.format(epoch))

        # evaluate f1
        dev_f1 = evaluate_model_on_f1(model, params, vocab, bucket_batcher, data.dev_instances)
        train_f1 = evaluate_model_on_f1(model, params, vocab, bucket_batcher, data.train_instances)
        dev_f1s.append(dev_f1)
        train_f1s.append(train_f1)
        sys.stdout.flush()

        # train
        model.train()
        train_generator = bucket_batcher(data.train_instances, num_epochs=1)
        for step, batch in enumerate(train_generator):
            batch['training'] = True
            loss = model.train_on_batch(batch, optimizer)
            # print
            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    # to pandas
    s1 = pd.Series(train_f1s, index=np.arange(params.max_epochs))
    s1.name = 'train_f1'
    s2 = pd.Series(dev_f1s, index=np.arange(params.max_epochs))
    s2.name = 'dev_f1'

    return [s1, s2]
