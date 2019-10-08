import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import time
import sys
import numpy as np
import pandas as pd
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator


from dlsrl.data import Data
from dlsrl.eval import f1_official_conll05
from dlsrl.models import make_model
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
        res = ''
        for k, v in sorted(self.param2val.items()):
            res += '{}={}\n'.format(k, v)
        return res


def main(param2val):

    if config.Global.local:
        print('WARNING: Loading data locally because config.Global.local=True')
        config.RemoteDirs = config.LocalDirs

    if config.LocalDirs.glove.exists():
        glove_path = config.LocalDirs.glove
    else:
        glove_path = config.RemoteDirs.glove

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # make local folder for saving checkpoint + events files
    local_job_p = config.LocalDirs.runs / params.job_name
    if not local_job_p.exists():
        local_job_p.mkdir(parents=True)

    # data + vocab
    print('Building data...')
    data = Data(params)
    vocab = Vocabulary.from_instances(data.train_instances + data.dev_instances)

    # batching
    bucket_batcher = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(vocab)

    # model
    print('Building model...')
    model = make_model(params, vocab, glove_path)

    # optimizer
    if params.my_implementation:
        optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                           epsilon=params.epsilon,
                                           clipnorm=params.max_grad_norm)
    else:
        optimizer = torch.optim.Adadelta(params=model.parameters(), lr=params.learning_rate, eps=params.epsilon)

    # train loop
    dev_f1s = []
    train_start = time.time()
    for epoch in range(params.max_epochs):
        print()
        print('===========')
        print('Epoch: {}'.format(epoch))
        print('===========')

        # ----------------------------------------------- start evaluation

        # conll05 evaluation data
        all_sentence_pred_labels_no_pad = []
        all_sentence_gold_labels_no_pad = []
        all_verb_indices = []
        all_sentences_no_pad = []

        model.eval()
        dev_generator = bucket_batcher(data.dev_instances, num_epochs=1)
        for step, batch in enumerate(dev_generator):

            if len(batch['tags']) != params.batch_size:
                print('WARNING: Batch size is {}. Skipping'.format(len(batch['tags'])))
                continue

            # get predictions
            batch['training'] = False
            output_dict = model(**batch)  # input is dict[str, tensor]
            softmax_3d = output_dict['softmax_3d']  # [mb_size, max_sent_length, num_labels]

            # get words and verb_indices
            sentences_b = []
            verb_indices_b = []
            for row in batch['metadata']:  # this is correct
                sentence = row['words']
                verb_index = row['verb_index']
                sentences_b.append(sentence)
                verb_indices_b.append(verb_index)

            # rename variables
            y_b = batch['tags'].cpu().numpy()
            x1_b = batch['tokens']['tokens'].cpu().numpy()
            x2_b = batch['verb_indicator'].cpu().numpy()

            # get gold and predicted label IDs
            batch_pred_label_ids = np.argmax(softmax_3d, axis=2)  # [batch_size, seq_length]
            batch_gold_label_ids = y_b  # [batch_size, seq_length]
            assert np.shape(batch_pred_label_ids) == (params.batch_size, np.shape(x1_b)[1])
            assert np.shape(batch_gold_label_ids) == (params.batch_size, np.shape(x1_b)[1])

            # collect data for evaluation
            for x1_row, x2_row, gold_label_ids, pred_label_ids, s, vi in zip(x1_b,
                                                                             x2_b,
                                                                             batch_gold_label_ids,
                                                                             batch_pred_label_ids,
                                                                             sentences_b,
                                                                             verb_indices_b):

                # convert IDs to tokens
                sentence_pred_labels = [vocab.get_token_from_index(i, namespace="labels")
                                        for i in pred_label_ids]
                sentence_gold_labels = [vocab.get_token_from_index(i, namespace="labels")
                                        for i in gold_label_ids]

                # collect data for conll-05 evaluation + remove padding
                sentence_length = len(s)
                all_sentence_pred_labels_no_pad.append(sentence_pred_labels[:sentence_length])
                all_sentence_gold_labels_no_pad.append(sentence_gold_labels[:sentence_length])
                all_verb_indices.append(vi)
                all_sentences_no_pad.append(s)

        for label in all_sentence_gold_labels_no_pad:
            assert label != config.Data.pad_label

        # evaluate with official conll05 perl script with Python interface provided by Allen AI NLP toolkit
        sys.stdout.flush()
        print('=============================================')
        print('Official Conll-05 Evaluation on Dev Split')
        dev_f1 = f1_official_conll05(all_sentence_pred_labels_no_pad,  # List[List[str]]
                                     all_sentence_gold_labels_no_pad,  # List[List[str]]
                                     all_verb_indices,  # List[Optional[int]]
                                     all_sentences_no_pad)  # List[List[str]]
        print('=============================================')
        sys.stdout.flush()

        # collect
        dev_f1s.append(dev_f1)

        # ----------------------------------------------- end evaluation

        # train on batches
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
