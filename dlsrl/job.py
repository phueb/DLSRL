import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import shutil
import time
import sys
from sklearn.metrics import f1_score
import numpy as np

from dlsrl.data import Data
from dlsrl.utils import get_batches, shuffle_stack_pad, count_zeros_from_end
from dlsrl.eval import print_f1, f1_naive, f1_official_conll05
from dlsrl.model import Model
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
    data = Data(params)  # TODO make tf.data.Dataset?

    # model
    deep_lstm = Model(params, data.embeddings, data.num_labels)

    optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                       epsilon=params.epsilon,
                                       clipnorm=params.max_grad_norm)

    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()  # performs softmax internally

    # train loop
    train_start = time.time()
    for epoch in range(params.max_epochs):
        print()
        print('====================================================')
        print('Epoch: {}'.format(epoch))

        # TODO save checkpoint from which to load model
        ckpt_p = local_job_p / "epoch_{}.ckpt".format(epoch)

        # prepare data for epoch
        train_x1, train_x2, train_y = shuffle_stack_pad(data.train,
                                                        batch_size=params.batch_size)  # returns int32
        dev_x1, dev_x2, dev_y = shuffle_stack_pad(data.dev,
                                                  batch_size=config.Eval.dev_batch_size,
                                                  shuffle=False)

        # ----------------------------------------------- start evaluation

        all_gold_label_ids_no_pad = []
        all_pred_label_ids_no_pad = []
        all_lengths = []

        # TODO test
        all_sentence_pred_labels_no_pad = []
        all_sentence_gold_labels_no_pad = []
        all_verb_indices = []  # TODO this is unused
        all_sentences = []

        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(dev_x1, dev_x2, dev_y, config.Eval.dev_batch_size)):

            softmax_probs = deep_lstm(x1_b, x2_b, training=False)

            batch_pred_label_ids_pad = np.argmax(softmax_probs, axis=1)  # [num_words]
            batch_gold_label_ids_pad = y_b.flatten()  # reshape from [batch-size, max_seq_len] to [num_words]
            batch_lengths = [len(row) - count_zeros_from_end(row) for row in x1_b]
            batch_sent_boundaries = np.cumsum(batch_lengths)
            batch_words_pad = [data.sorted_words[i] for i in x1_b.flatten()]

            all_lengths += batch_lengths

            batch_gold_label_ids_no_pad = []
            batch_pred_label_ids_no_pad = []
            batch_words = []

            # remove padding (but not "O" labels for words that are "outside" arguments)
            for g, p, w in zip(batch_gold_label_ids_pad,
                               batch_pred_label_ids_pad,
                               batch_words_pad):
                if g != 0:
                    # batch
                    batch_gold_label_ids_no_pad.append(g)
                    batch_pred_label_ids_no_pad.append(p)
                    batch_words.append(w)
                    # all
                    all_gold_label_ids_no_pad.append(g)
                    all_pred_label_ids_no_pad.append(p)
                    if config.Eval.verbose:
                        print('gold={:<3} pred={:<3}'.format(g, p))

            # need to collect data for conll05 evaluation script
            batch_gold_labels_no_pad = np.array([data.sorted_labels[i] for i in all_gold_label_ids_no_pad])
            batch_pred_labels_no_pad = np.array([data.sorted_labels[i] for i in all_pred_label_ids_no_pad])
            batch_sentence_gold_labels_no_pad = np.split(batch_gold_labels_no_pad, batch_sent_boundaries)[:-1]
            batch_sentence_pred_labels_no_pad = np.split(batch_pred_labels_no_pad, batch_sent_boundaries)[:-1]

            batch_verb_indices = list(np.argmax(x2_b, axis=1))
            assert len(batch_verb_indices) == config.Eval.dev_batch_size

            batch_sentences = np.split(np.array(batch_words), batch_sent_boundaries)[:-1]
            assert len(batch_sentences) == config.Eval.dev_batch_size

            all_sentence_gold_labels_no_pad += batch_sentence_gold_labels_no_pad
            all_sentence_pred_labels_no_pad += batch_sentence_pred_labels_no_pad
            all_verb_indices += batch_verb_indices
            all_sentences += batch_sentences

            # check that verb indices are not bigger than sentence length
            for vi, s in zip(batch_verb_indices, batch_sentences):
                assert vi <= len(s)

        num_dev_labels = len(all_gold_label_ids_no_pad)
        num_dev_sentences = len(all_lengths)
        print('Number of labels to evaluate after excluding PAD_LABEL={}'.format(num_dev_labels))
        print('Number of sentences to evaluate after excluding PAD_LABEL={}'.format(num_dev_sentences))

        # evaluate f1 score computed over single labels (not spans)
        # f1_score expects 1D label ids (e.g. gold=[0, 2, 1, 0], pred=[0, 1, 1, 0])
        print_f1(epoch, 'weight', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='weighted'))
        print_f1(epoch, 'macro ', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='macro'))
        print_f1(epoch, 'micro ', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='micro'))

        # evaluate with pseudo-conll05 Python function
        print_f1(epoch, 'conll-05', f1_naive(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, all_lengths))

        # evaluate with official conll05 perl script with Python interface provided by Allen AI NLP toolkit
        print('===+ Official Conll-05 Evaluation on Dev Split +===')
        f1_official_conll05(all_sentence_pred_labels_no_pad,        # List[List[str]]
                            all_sentence_gold_labels_no_pad,        # List[List[str]]
                            all_verb_indices,                       # List[Optional[int]]
                            all_sentences)                          # List[List[str]]
        print('===+ Official Conll-05 Evaluation on Dev Split +===')

        # ----------------------------------------------- end evaluation

        # train on batches
        for step, (x1_b, x2_b, y_b) in enumerate(get_batches(train_x1, train_x2, train_y, params.batch_size)):

            with tf.GradientTape() as tape:
                softmax_probs = deep_lstm(x1_b, x2_b, training=True)  # returns [num_words, num_labels]
                loss = cross_entropy(y_b.reshape([-1]), softmax_probs)
                # no need to mask loss function because padding is masked, and "O" labels should be learned

            grads = tape.gradient(loss, deep_lstm.trainable_weights)
            optimizer.apply_gradients(zip(grads, deep_lstm.trainable_weights))

            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss = {:2.2f} minutes elapsed = {:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    #  move events file to shared drive
    events_p = list(local_job_p.glob('*events*'))[0]
    dst = config.RemoteDirs.runs / params.param_name / params.job_name
    if not dst.exists():
        dst.mkdir(parents=True)
    shutil.move(str(events_p), str(dst))


