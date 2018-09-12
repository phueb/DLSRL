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

# get config
with open(CONFIG_FILE_PATH, 'r') as config_file:
    config = json.load(config_file, object_hook=lambda d: Namespace(**d))

# get data
train_sents, dev_sents, word_dict, label_dict, embeddings = get_srl_data(
    config, TRAIN_DATA_PATH, DEV_DATA_PATH, VOCAB_PATH, LABEL_PATH)
data = TaggerData(config, train_sents, dev_sents, word_dict, label_dict, embeddings)

# model
model = Model(config, data)  # TODO

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
global_step = 0
epoch = 0
train_loss = 0.0
while epoch < config.max_epochs:
    for x_batch, y_batch, mask_batch in data.get_batched_tensors():  # TODO use mask?

        # print(x_batch)
        # print(y_batch)
        # print(mask_batch)
        # print()

        # x is list of arrays with dim [sent_len, 2]
        # y is list of vectors with dim [sent_len]
        # mask is list of vectors with dim [sent_len] informing about length of sent


        feed_dict = {model.x: x_batch,
                     model.y: y_batch}
        loss = sess.run(model.tf_loss, feed_dict=feed_dict)
        train_loss += loss
        i += 1
        global_step += 1
        if i % PRINT_INTERVAL == 0:
            print("{} training steps, loss={:.3f}".format(i, train_loss / i))

    train_loss = train_loss / i
    print("Epoch {}, steps={}, loss={:.3f}".format(epoch, i, train_loss))
    i = 0
    epoch += 1
    train_loss = 0.0

