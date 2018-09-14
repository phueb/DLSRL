#!/usr/bin/env/python3

import json
from argparse import Namespace
import tensorflow as tf
import time
from tfvis import Timeline
import argparse
from pycm import ConfusionMatrix
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_utils import get_data
from batcher import Batcher
from model import Model

TRAIN_DATA_PATH = 'data/conll05.train.txt'
DEV_DATA_PATH =  'data/conll05.dev.txt'
VOCAB_PATH = None
LABEL_PATH = None

EVALUATE_WHICH = ['dev']
LOSS_INTERVAL = 100
TFVIS_PATH = None  #'/home/ph'


# TODO use mask to prevent training on zeros?


def srl_task(config_file_path):
    def evaluate(which):
        batch_predictions = []
        batch_actuals = []
        for w_ids_batch, p_ids_batch, l_ids_batch in batcher.get_batched_tensors(which=which):
            feed_dict = {model.word_ids: w_ids_batch,
                         model.predicate_ids: p_ids_batch,
                         model.label_ids: l_ids_batch}
            batch_pred = sess.run(model.predictions, feed_dict=feed_dict)
            batch_predictions.append(batch_pred.flatten())
            batch_actuals.append(l_ids_batch.flatten())
        # confusion matrix based metrics
        a = np.concatenate(batch_actuals, axis=0)
        p = np.concatenate(batch_predictions, axis=0)
        nonzero_ids = np.nonzero(a)
        actual = a[nonzero_ids]  # TODO remove zeros? i think zero is a label!
        predicted = p[nonzero_ids]
        cm = ConfusionMatrix(actual_vector=actual, predict_vector=predicted)
        print('/////////////////////////// EVALUATION')
        ppvs = np.array([i for i in cm.PPV.values() if i != 'None']).astype(np.float)
        tprs = np.array([i for i in cm.TPR.values() if i != 'None']).astype(np.float)
        f1s = np.array([i for i in cm.F1.values() if i != 'None']).astype(np.float)
        print('num total labels={} num nonzero labels={}'.format(len(a), len(actual)))
        print('PYMC    | precision={:.3f} recall={:.3f} f1={:.3f}'.format(
            np.mean(ppvs),
            np.mean(tprs),
            np.mean(f1s)))  # TODO how to get aggregate f1 score?
        print('sklearn | precision={:.3f} recall={:.3f} f1={:.3f}'.format(
            f1_score(a, p, average='macro'),
            precision_score(a, p, average="macro"),
            recall_score(a, p, average="macro")))

        # config
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file, object_hook=lambda d: Namespace(**d))
    print('///// Configs START')
    for k in sorted(config.__dict__):
        v = vars(config)[k]
        print('    {:>20}={:<20}'.format(k, v))
    print('///// Configs END')

    # data
    train_data, dev_data, word_dict, num_labels, embeddings = get_data(
        config, TRAIN_DATA_PATH, DEV_DATA_PATH, VOCAB_PATH, LABEL_PATH)

    # model
    batcher = Batcher(config, train_data, dev_data)
    model = Model(config, embeddings, num_labels)

    # profiling
    if TFVIS_PATH:
        with tf.train.MonitoredSession() as sess:
            with Timeline() as timeline:
                w_ids_batch, p_ids_batch, l_ids_batch = next(batcher.get_batched_tensors('train'))
                feed_dict = {model.word_ids: w_ids_batch,
                             model.predicate_ids: p_ids_batch,
                             model.label_ids: l_ids_batch}
                sess.run(model.update, feed_dict=feed_dict, **timeline.kwargs)
                print('visualizing timeline at {}...'.format(TFVIS_PATH))
                timeline.visualize("{}/profile_num_layers={}.html".format(TFVIS_PATH, config.num_layers))

    # train
    local_step = 0
    global_step = 0
    train_loss = 0.0
    global_start = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.max_epochs):

            # eval
            for which_data in EVALUATE_WHICH:
                evaluate(which_data)

            # train
            epoch_start = time.time()
            for w_ids_batch, p_ids_batch, l_ids_batch in batcher.get_batched_tensors('train'):
                feed_dict = {model.word_ids: w_ids_batch,
                             model.predicate_ids: p_ids_batch,
                             model.label_ids: l_ids_batch}
                loss, _ = sess.run([model.mean_loss, model.update],
                                   feed_dict=feed_dict)
                train_loss += loss
                local_step += 1
                global_step += 1
                if local_step % LOSS_INTERVAL == 0 or local_step == 1:
                    print("step {:>6} (global {:>6}): loss={:.3f}, epoch min={:.3f} total min={:.3f}".format(
                        local_step,
                        global_step,
                        train_loss / local_step,
                        (time.time() - epoch_start) / 60,
                        (time.time() - global_start) / 60))

            print("Completed Epoch {:3}\n".format(epoch))
            local_step = 0
            train_loss = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store", default=1, dest='config_number', type=int,
                        help='Number of config_file (e.g. "1" in config1.json)')
    namespace = parser.parse_args()
    config_file_path = 'configs/' + 'config{}.json'.format(namespace.config_number)
    srl_task(config_file_path)