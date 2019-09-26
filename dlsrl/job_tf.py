import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import time
import sys
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

from dlsrl.data import Data
from dlsrl.utils import get_batches, shuffle_stack_pad, count_zeros_from_end
from dlsrl.eval import print_f1, f1_official_conll05
from dlsrl.model_tf import TensorflowSRLModel
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


def masked_sparse_categorical_crossentropy(y_true, y_pred, mask_value=0):
    mask_value = tf.Variable(mask_value)
    mask = tf.equal(y_true, mask_value)
    mask = 1 - tf.cast(mask, tf.float32)

    # multiply categorical_crossentropy with the mask  (no reduce_sum operation is performed)
    _loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return tf.math.reduce_sum(_loss) / tf.math.reduce_sum(mask)


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
    data = Data(params)  # TODO make tf.data.Dataset? + use smarter batching function to reduce amount of padding

    # model
    model_tf = TensorflowSRLModel(params, data.embeddings, data.num_labels)

    optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                       epsilon=params.epsilon,
                                       clipnorm=params.max_grad_norm)

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # train loop
    dev_f1s = []
    train_start = time.time()
    for epoch in range(params.max_epochs):
        print()
        print('===========')
        print('Epoch: {}'.format(epoch))
        print('===========')

        # TODO save checkpoint from which to load model
        ckpt_p = local_job_p / "epoch_{}.ckpt".format(epoch)

        # prepare data for epoch
        train_x1, train_x2, train_y = shuffle_stack_pad(data.train,
                                                        batch_size=params.batch_size)  # returns int32
        dev_x1, dev_x2, dev_y = shuffle_stack_pad(data.dev,
                                                  batch_size=config.Eval.dev_batch_size,
                                                  shuffle=False)

        # ----------------------------------------------- start evaluation

        # per-label f1 evaluation
        all_gold_label_ids_no_pad = []
        all_pred_label_ids_no_pad = []

        # conll05 evaluation data
        all_sentence_pred_labels_no_pad = []
        all_sentence_gold_labels_no_pad = []
        all_verb_indices = []
        all_sentences_no_pad = []

        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(dev_x1, dev_x2, dev_y, config.Eval.dev_batch_size)):

            # get predicted label_ids from model
            softmax_2d = model_tf(x1_b, x2_b, training=False)  # [num words in batch, num_labels]
            softmax_3d = np.reshape(softmax_2d, (*np.shape(x1_b), data.num_labels))  # 1st dim is batch_size
            batch_pred_label_ids = np.argmax(softmax_3d, axis=2)  # [batch_size, seq_length]
            batch_gold_label_ids = y_b  # [batch_size, seq_length]
            assert np.shape(batch_pred_label_ids) == (config.Eval.dev_batch_size, np.shape(x1_b)[1])

            # collect data for evaluation
            for x1_row, x2_row, gold_label_ids, pred_label_ids, in zip(x1_b,
                                                                       x2_b,
                                                                       batch_gold_label_ids,
                                                                       batch_pred_label_ids):

                sentence_length = len(x1_row) - count_zeros_from_end(x1_row)

                assert count_zeros_from_end(x1_row) == count_zeros_from_end(gold_label_ids)
                sentence_gold_labels = [data.sorted_labels[i] for i in gold_label_ids]
                sentence_pred_labels = [data.sorted_labels[i] for i in pred_label_ids]
                verb_index = np.argmax(x2_row)
                sentence = [data.sorted_words[i] for i in x1_row]

                # collect data for conll-05 evaluation + remove padding
                all_sentence_pred_labels_no_pad.append(sentence_pred_labels[:sentence_length])
                all_sentence_gold_labels_no_pad.append(sentence_gold_labels[:sentence_length])
                all_verb_indices.append(verb_index)
                all_sentences_no_pad.append(sentence[:sentence_length])

                # collect data for per-label evaluation
                all_gold_label_ids_no_pad += list(gold_label_ids[:sentence_length])
                all_pred_label_ids_no_pad += list(pred_label_ids[:sentence_length])

        print('Number of sentences to evaluate: {}'.format(len(all_sentences_no_pad)))

        for label in all_sentence_gold_labels_no_pad:
            assert label != config.Data.pad_label

        # evaluate f1 score computed over single labels (not spans)
        # f1_score expects 1D label ids (e.g. gold=[0, 2, 1, 0], pred=[0, 1, 1, 0])
        print_f1(epoch, 'weight', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='weighted'))
        print_f1(epoch, 'macro ', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='macro'))
        print_f1(epoch, 'micro ', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='micro'))

        # evaluate with official conll05 perl script with Python interface provided by Allen AI NLP toolkit
        sys.stdout.flush()
        print('=============================================')
        print('Official Conll-05 Evaluation on Dev Split')
        dev_f1 = f1_official_conll05(all_sentence_pred_labels_no_pad,  # List[List[str]]
                                     all_sentence_gold_labels_no_pad,  # List[List[str]]
                                     all_verb_indices,  # List[Optional[int]]
                                     all_sentences_no_pad)  # List[List[str]]
        print_f1(epoch, 'conll-05', dev_f1)
        dev_f1s.append(dev_f1)
        print('=============================================')
        sys.stdout.flush()

        # ----------------------------------------------- end evaluation

        # train on batches
        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(train_x1, train_x2, train_y, params.batch_size)):

            with tf.GradientTape() as tape:
                softmax_2d = model_tf(x1_b, x2_b, training=True)  # returns [num words in batch, num_labels]
                y_true = y_b.reshape([-1])
                y_pred = softmax_2d
                # loss = cross_entropy(y_true=y_true,
                #                      y_pred=y_pred)

                # this masked loss function is different than above and does not decrease as fast
                loss = masked_sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)

            grads = tape.gradient(loss, model_tf.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_tf.trainable_weights))

            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    # to pandas
    eval_epochs = np.arange(params.max_epochs)
    df_dev_f1 = pd.DataFrame(dev_f1s, index=eval_epochs, columns=['dev_f1'])
    df_dev_f1.name = 'dev_f1'

    return [df_dev_f1]
