from dotenv import load_dotenv
import json
from argparse import Namespace
import tensorflow as tf
import time
import argparse
import socket

from data_utils import get_data
from train_utils import get_feed_dicts, evaluate, shuffle_stack_pad
from model import Model

TRAIN_DATA_PATH = 'data/conll05.train.txt'
DEV_DATA_PATH =  'data/conll05.dev.txt'

TENSORBOARD_DIR = 'tb'

LOSS_INTERVAL = 100


def export_to_tensorboard(model, sess, feed_dict, global_step):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
    summary = sess.run(model.merged1,
                       feed_dict=feed_dict,
                       options=run_options)
    model.train_writer.add_summary(summary, global_step)


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
    train_data, dev_data, word_dict, label_dict, embeddings = get_data(
        config, TRAIN_DATA_PATH, DEV_DATA_PATH)

    epoch_step = 0
    global_step = 0
    epoch_loss_sum = 0.0
    global_start = time.time()
    g = tf.Graph()
    with g.as_default():
        model = Model(config, embeddings, label_dict.size(), g)
        sess = tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True,
                                                         log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.max_epochs):
            # eval
            evaluate(dev_data, model, sess, epoch, global_step, word_dict, label_dict)
            # train
            x1, x2, y = shuffle_stack_pad(train_data, config.train_batch_size)
            epoch_start = time.time()
            for feed_dict in get_feed_dicts(x1, x2, y, model, config.train_batch_size, config.keep_prob):
                if epoch_step % LOSS_INTERVAL == 0:
                    export_to_tensorboard(model, sess, feed_dict, global_step)
                    print("step {:>6} epoch {:>3}: loss={:1.3f}, epoch sec={:3.0f}, total hrs={:.1f}".format(
                        epoch_step,
                        epoch,
                        epoch_loss_sum / max(epoch_step, 1),
                        (time.time() - epoch_start),
                        (time.time() - global_start) / 3600))
                loss, _ = sess.run([model.nonzero_mean_loss, model.update], feed_dict=feed_dict)
                epoch_loss_sum += loss
                epoch_step += 1
                global_step += 1
            epoch_step = 0
            epoch_loss_sum = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store", default=0, dest='config_number', type=int,
                        help='Number of config_file (e.g. "1" in config1.json)')
    namespace = parser.parse_args()
    config_file_path = 'configs/' + 'config{}.json'.format(namespace.config_number)
    if socket.gethostname() == 'Ursa':
        p = 'Ursa.env'
    else:
        p = 'not_Ursa.env'
        load_dotenv(dotenv_path=p, verbose=True, override=True)
    srl_task(config_file_path)