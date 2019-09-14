import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import shutil
import time
import sys
import numpy as np

from dlsrl.data import Data
from dlsrl.utils import get_batches, shuffle_stack_pad, count_zeros_from_end, make_word2embed
from dlsrl.eval import evaluate
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
    word2embed = make_word2embed(params)
    data = Data(params, word2embed)

    # train loop
    train_start = time.time()

    # model
    deep_lstm = Model(params, data.embeddings, data.num_labels)
    optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                       epsilon=params.epsilon)

    # TODO eval before training

    loss_metric = tf.keras.metrics.Mean()
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()  # performs softmax internally
    for epoch in range(params.max_epochs):
        print('Epoch: {}'.format(epoch))

        # TODO save checkpoint from which to load model
        ckpt_p = local_job_p / "epoch_{}.ckpt".format(epoch)

        x1, x2, y = shuffle_stack_pad(train_data, params.batch_size)
        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(x1, x2, y, params.batch_size)):

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

            loss_metric(loss)
            if step % 1 == 0:
                print('step {:<6}: mean loss = {:2.2f} minutes elapsed = {:<3}'.format(
                    step, loss_metric.result(), (time.time() - train_start) // 60))

        raise SystemExit('Completed first epoch')

        # TODO evaluate
        evaluate(dev_data, deep_lstm, epoch)








    #  move events file to shared drive
    events_p = list(local_job_p.glob('*events*'))[0]
    dst = config.RemoteDirs.runs / params.param_name / params.job_name
    if not dst.exists():
        dst.mkdir(parents=True)
    shutil.move(str(events_p), str(dst))


