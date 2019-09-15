import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import shutil
import time
import sys
from sklearn.metrics import f1_score
import numpy as np

from dlsrl.data import Data
from dlsrl.utils import get_batches, shuffle_stack_pad, count_zeros_from_end
from dlsrl.eval import print_f1
from dlsrl.model import Model
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

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # make local folder for saving checkpoint + events files
    local_job_p = config.LocalDirs.runs / params.job_name
    if not local_job_p.exists():
        local_job_p.mkdir(parents=True)

    # data
    data = Data(params)

    # model
    deep_lstm = Model(params, data.embeddings, data.num_labels)
    optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                       epsilon=params.epsilon)

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()  # performs softmax internally

    # train loop
    train_start = time.time()
    for epoch in range(params.max_epochs):
        print('Epoch: {}'.format(epoch))

        # TODO save checkpoint from which to load model
        ckpt_p = local_job_p / "epoch_{}.ckpt".format(epoch)

        # prepare data for epoch
        train_x1, train_x2, train_y = shuffle_stack_pad(data.train_data,
                                                        batch_size=params.batch_size)  # returns int32
        dev_x1, dev_x2, dev_y = shuffle_stack_pad(data.dev_data,
                                                  batch_size=config.Eval.dev_batch_size,
                                                  shuffle=False)

        # TODO use tensorflow f1 metric

        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(dev_x1, dev_x2, dev_y, config.Eval.dev_batch_size)):

            mask = np.clip(x1_b, 0, 1)

            pred_label_ids = deep_lstm.predict_label_ids(x1_b, x2_b, mask)
            gold_label_ids = y_b.flatten()

            print(pred_label_ids[:10])
            print(pred_label_ids.shape)
            print(gold_label_ids[:10])
            print(gold_label_ids.shape)  # TODO shape is way too big

            # f1_score expects 1D label ids (e.g. gold=[0, 2, 1, 0], pred=[0, 1, 1, 0])
            print_f1(epoch, 'average=weight', f1_score(gold_label_ids, pred_label_ids, average='weighted'))
            print_f1(epoch, 'average=macro ', f1_score(gold_label_ids, pred_label_ids, average='macro'))
            print_f1(epoch, 'average=micro ', f1_score(gold_label_ids, pred_label_ids, average='micro'))

        # train on batches
        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(train_x1, train_x2, train_y, params.batch_size)):

            # pre-processing  # TODO use tf.data
            lengths = [len(row) - count_zeros_from_end(row) for row in x1_b]
            max_seq_len = np.max(lengths)
            word_ids = x1_b[:, :max_seq_len]
            predicate_ids = x2_b[:, :max_seq_len]
            mask = np.clip(x1_b, 0, 1)
            flat_label_ids = y_b[:, :max_seq_len].reshape([-1])

            with tf.GradientTape() as tape:
                logits = deep_lstm(word_ids, predicate_ids, mask)
                loss = cross_entropy(flat_label_ids, logits)  # TODO mask loss function?

            grads = tape.gradient(loss, deep_lstm.trainable_weights)
            optimizer.apply_gradients(zip(grads, deep_lstm.trainable_weights))
            # TODO implement gradient clipping by max norm

            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss = {:2.2f} minutes elapsed = {:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))



            # print_f1(epoch, 'args-exclude1', f1_conll05(gold, pred, lengths, True))
            # print_f1(epoch, 'args-include1', f1_conll05(gold, pred, lengths, False))

    #  move events file to shared drive
    events_p = list(local_job_p.glob('*events*'))[0]
    dst = config.RemoteDirs.runs / params.param_name / params.job_name
    if not dst.exists():
        dst.mkdir(parents=True)
    shutil.move(str(events_p), str(dst))


