import json
from argparse import Namespace
import tensorflow as tf
import time

from data_utils import get_data
from batcher import Batcher
from model import Model

CONFIG_FILE_PATH = 'srl_small_config.json'
TRAIN_DATA_PATH = 'conll05.test.wsj.txt'
DEV_DATA_PATH = 'sample_sentences.txt'  # TODO change
VOCAB_PATH = None
LABEL_PATH = None

LOSS_INTERVAL = 10

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
    for w_ids_batch, p_ids_batch, l_ids_batch, mask_batch in batcher.get_batched_tensors():  # TODO use mask?
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

