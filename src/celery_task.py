import json
import tensorflow as tf
import time
import argparse
from pathlib import Path
import socket
import os

from src.data_utils import get_data
from src.train_utils import get_batches, evaluate, shuffle_stack_pad, make_feed_dict
from src.model import Model
from src import config_str

TRAIN_DATA_PATH = 'data/conll05.train.txt'
DEV_DATA_PATH = 'data/conll05.dev.txt'

LOSS_INTERVAL = 100


def srl_task(**kwargs):

    # make config
    d = json.loads(config_str)
    for k, v in kwargs.items():
        d[k] = v
    config = argparse.Namespace(**d)
    print('///// Configs START')
    for k, v in sorted(d.items()):
        print('    {:>20}={:<20}'.format(k, v))
    print('///// Configs END')

    # save config
    hostname = socket.gethostname()
    jscon_configs_dir = Path(os.environ['JSON_CONFIGS_DIR'])
    if not jscon_configs_dir.is_dir():
        jscon_configs_dir.mkdir()
    json.dump(d, (jscon_configs_dir / 'config_{}.json'.format(hostname)).open('w'), ensure_ascii=False)
    print('Saved configs to {}'.format(jscon_configs_dir))

    # data
    train_data, dev_data, word_dict, label_dict, embeddings = get_data(
        config, TRAIN_DATA_PATH, DEV_DATA_PATH)

    # train loop
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
            evaluate(dev_data, model, sess, epoch, global_step)
            x1, x2, y = shuffle_stack_pad(train_data, config.train_batch_size)
            epoch_start = time.time()
            for x1_b, x2_b, y_b in get_batches(x1, x2, y, config.train_batch_size):
                feed_dict = make_feed_dict(x1_b, x2_b, y_b, model, config.keep_prob)
                if epoch_step % LOSS_INTERVAL == 0:
                    # tensorboard
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                    scalar_summaries = sess.run(model.scalar_summaries,
                                       feed_dict=feed_dict,
                                       options=run_options)
                    model.train_writer.add_summary(scalar_summaries, global_step)
                    # print info
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