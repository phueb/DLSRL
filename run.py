import json
from argparse import Namespace
import tensorflow as tf
import time

from data_utils import get_data
from batcher import Batcher
from model import Model

CONFIG_FILE_PATH = 'config.json'
TRAIN_DATA_PATH = 'data/conll05.train.txt'
DEV_DATA_PATH =  'data/conll05.dev.txt'
VOCAB_PATH = None
LABEL_PATH = None

LOSS_INTERVAL = 100

# TODO
# use python library to calculate confusion matrix evaluations

# config
with open(CONFIG_FILE_PATH, 'r') as config_file:
    config = json.load(config_file, object_hook=lambda d: Namespace(**d))

# data
train_data, dev_data, word_dict, num_labels, embeddings = get_data(
    config, TRAIN_DATA_PATH, DEV_DATA_PATH, VOCAB_PATH, LABEL_PATH)

# model
batcher = Batcher(config, train_data, dev_data)
model = Model(config, embeddings, num_labels)

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
global_step = 0
epoch = 0
train_loss = 0.0
start = time.time()
while epoch < config.max_epochs:

    # eval on dev
    dev_accuracies = []
    for w_ids_batch, p_ids_batch, l_ids_batch in batcher.get_batched_tensors('dev'):
        feed_dict = {model.word_ids: w_ids_batch,
                     model.predicate_ids: p_ids_batch,
                     model.label_ids: l_ids_batch}
        [dev_accuracy] = sess.run([model.accuracy], feed_dict=feed_dict)
        dev_accuracies.append(dev_accuracy)
    print('Dev accuracy={:.3f}'.format(sum(dev_accuracies) / len(dev_accuracies)))

    # train
    for w_ids_batch, p_ids_batch, l_ids_batch in batcher.get_batched_tensors('train'):
        feed_dict = {model.word_ids: w_ids_batch,
                     model.predicate_ids: p_ids_batch,
                     model.label_ids: l_ids_batch}
        loss, _ = sess.run([model.mean_loss, model.update], feed_dict=feed_dict)
        train_loss += loss
        i += 1
        global_step += 1
        if i % LOSS_INTERVAL == 0 or i == 1:
            print("step {:>3}: loss={:.3f}, min elapsed={:.1f}".format(i, train_loss / i, (time.time() - start) / 60))

    print("Completed Epoch {:3}\n".format(epoch))
    i = 0
    epoch += 1
    train_loss = 0.0
    



    

