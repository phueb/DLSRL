import json
import tensorflow as tf
import time
import argparse
from pathlib import Path
import socket

from src.data_utils import get_data
from src.train_utils import get_feed_dicts, evaluate, shuffle_stack_pad
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
    print('///// Configs START')
    for k, v in sorted(d.items()):
        print('    {:>20}={:<20}'.format(k, v))
    print('///// Configs END')

    # save config
    hostname = socket.gethostname()
    dir = Path(__file__).parent.parent / 'configs'
    if not dir.is_dir():
        dir.mkdir()
    json.dump(d, (dir / 'config_{}.json'.format(hostname)).open('w'), ensure_ascii=False)
    print('Saved configs to {}'.format(dir))

    config = argparse.Namespace(**d)

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
                    # tensorboard
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                    summary = sess.run(model.merged1,
                                       feed_dict=feed_dict,
                                       options=run_options)
                    model.train_writer.add_summary(summary, global_step)
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