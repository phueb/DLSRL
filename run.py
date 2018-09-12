import json
from argparse import Namespace
import tensorflow as tf

from data import get_srl_data
from tagger_data import TaggerData
from model import Model

CONFIG_FILE_PATH = 'srl_small_config.json'
TRAIN_DATA_PATH = 'sample_sentences.txt'
DEV_DATA_PATH = 'sample_sentences.txt'  # TODO change
VOCAB_PATH = None
LABEL_PATH = None

NUM_LAYERS = 2
NUM_UNITS = 300

PRINT_INTERVAL = 1

# config
with open(CONFIG_FILE_PATH, 'r') as config_file:
    config = json.load(config_file, object_hook=lambda d: Namespace(**d))

# data
train_sents, dev_sents, word_dict, label_dict, embeddings = get_srl_data(
    config, TRAIN_DATA_PATH, DEV_DATA_PATH, VOCAB_PATH, LABEL_PATH)
data = TaggerData(config, train_sents, dev_sents, word_dict, label_dict, embeddings)

# model
model = Model(config, data)

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
global_step = 0
epoch = 0
train_loss = 0.0
while epoch < config.max_epochs:
    for word_ids_batch, f_ids_batch, y_batch, mask_batch in data.get_batched_tensors():  # TODO use mask?
        feed_dict = {model.word_ids: word_ids_batch,
                     model.feature_ids: f_ids_batch,
                     model.y: y_batch}
        loss = sess.run(model.mean_loss, feed_dict=feed_dict)
        train_loss += loss
        i += 1
        global_step += 1
        if i % PRINT_INTERVAL == 0:
            print("{} training steps, loss={}".format(i, train_loss / i))

    train_loss = train_loss / i
    print("Epoch {}, steps={}, loss={}".format(epoch, i, train_loss))
    i = 0
    epoch += 1
    train_loss = 0.0

