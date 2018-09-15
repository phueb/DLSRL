# #!/usr/bin/env/python3

import json
from argparse import Namespace
import tensorflow as tf
import time
from tfvis import Timeline
import argparse

from data_utils import get_data
from train_utils import get_batches, evaluate
from model import Model

TRAIN_DATA_PATH = 'data/conll05.train.txt'
DEV_DATA_PATH =  'data/conll05.dev.txt'

LOSS_INTERVAL = 100
TFVIS_PATH = None  #'/home/ph'


# TODO use mask to prevent training on zeros?


def srl_task(config_file_path):

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
        config, TRAIN_DATA_PATH, DEV_DATA_PATH)

    # model
    model = Model(config, embeddings, num_labels)  # TODO num_labels

    # profiling
    if TFVIS_PATH:
        with tf.train.MonitoredSession() as sess:
            with Timeline() as timeline:
                feed_dict = next(get_batches(model, train_data, config.train_batch_size))
                sess.run(model.update, feed_dict=feed_dict, **timeline.kwargs)
                print('visualizing timeline at {}...'.format(TFVIS_PATH))
                timeline.visualize("{}/profile_num_layers={}.html".format(TFVIS_PATH, config.num_layers))

    # train
    local_step = 0
    train_loss = 0.0
    global_start = time.time()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.max_epochs):

            # eval
            evaluate(config, model, sess, dev_data)

            # train
            epoch_start = time.time()
            for feed_dict in get_batches(model, train_data, config.train_batch_size):
                loss, _ = sess.run([model.mean_loss, model.update],
                                   feed_dict=feed_dict)
                train_loss += loss
                local_step += 1
                if local_step % LOSS_INTERVAL == 0 or local_step == 1:
                    print("step {:>6} epoch {:>3}: loss={:.3f}, epoch min={:.3f} total min={:.3f}".format(
                        local_step,
                        epoch,
                        train_loss / local_step,
                        (time.time() - epoch_start) / 60,
                        (time.time() - global_start) / 60))

            local_step = 0
            train_loss = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store", default=0, dest='config_number', type=int,
                        help='Number of config_file (e.g. "1" in config1.json)')
    namespace = parser.parse_args()
    config_file_path = 'configs/' + 'config{}.json'.format(namespace.config_number)
    srl_task(config_file_path)