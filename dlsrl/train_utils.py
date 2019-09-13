import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def shuffle_stack_pad(data, batch_size, shuffle=True):
    """
    :param shuffle: whether to shuffle data
    :param batch_size: size batch to use
    :param data: [list of word_id seqs, list of pred_id seqs, list of label_id seqs]
    :return: zero-padded matrices for each list of shape [num_seqs, max_seq_len]
    """
    shape0 = len(data[1])
    num_excluded = shape0 % batch_size
    print('Excluding {} sequences due to fixed batch size'.format(num_excluded))
    shape0_adj = shape0 - num_excluded
    shape1 = np.max([len(i) for i in data[1]])
    mats = [np.zeros((shape0_adj, shape1)).astype(np.int32) for _ in range(3)]
    if shuffle:
        row_ids = np.random.choice(shape0, shape0_adj, replace=False)
    else:
        row_ids = np.arange(shape0_adj)
    for sequences, mat in zip(data, mats):
        for n, rand_id in enumerate(row_ids):
            seq = sequences[rand_id]
            mat[n, :len(seq)] = seq
    return mats  # x1, x2, y


def count_zeros_from_end(s):
    res = 0
    it = iter(s[::-1])
    while next(it) == 0:
        res += 1
    else:
        return res


def get_batches(x1, x2, y, batch_size):
    num_batches = len(x1) // batch_size
    print('Generating {} batches with size {}'.format(num_batches, batch_size))
    for x1_b, x2_b, y_b in zip(np.vsplit(x1, num_batches),
                               np.vsplit(x2, num_batches),
                               np.vsplit(y, num_batches)):

        yield x1_b, x2_b, y_b


def make_feed_dict(x1, x2, y, model, keep_prob):  # TODO remove this function for tensorflow v2.0
    lengths = [len(s) - count_zeros_from_end(s) for s in x1]
    max_seq_len = np.max(lengths)
    feed_dict = {model.word_ids: x1[:, :max_seq_len],
                 model.predicate_ids: x2[:, :max_seq_len],
                 model.label_ids: y[:, :max_seq_len],
                 model.keep_prob: keep_prob,
                 model.lengths: lengths}
    return feed_dict


def evaluate(data, model, summary_writer, sess, epoch, global_step):
    # make dev CONLL05
    batch_size = len(data[0])
    assert batch_size <= 4096  # careful with large CONLL05
    x1, x2, y = shuffle_stack_pad(data, batch_size=batch_size, shuffle=False)
    feed_dict = make_feed_dict(x1, x2, y, model, keep_prob=1.0)

    # TODO export confusion matrix - cm_summary

    # get predictions  # TODO viterbi decoding? add decoding constraint?
    pred, gold = sess.run([model.nonzero_predicted_label_ids, model.nonzero_label_ids_flat], feed_dict=feed_dict)

    # what is model predicting?
    for i, j in zip(gold[:100], pred[:100]):
        print('gold label="{}", predicted label="{}"'.format(i, j))

    # calc f1
    print('num labels={:,}'.format(len(gold)))
    print_f1(epoch, 'macro-labels', f1_score(gold, pred, average='macro'))
    print_f1(epoch, 'micro-labels', f1_score(gold, pred, average='micro'))
    print_f1(epoch, 'args-exclude1', f1_conll05(gold, pred, feed_dict[model.lengths], True))
    print_f1(epoch, 'args-include1', f1_conll05(gold, pred, feed_dict[model.lengths], False))


def print_f1(epoch, method, f1):
    print('epoch {:>3} method={} | f1={:.2f}'.format(epoch, method, f1))


def f1_conll05(gold, pred, lengths, exclude_label_one=True):  # TODO due to BIO tagging complete arguments are not compared (only B-to-B and I-to_I)
    hits = 1
    over_predictions = 1
    misses = 1
    start_p = 0
    for l in lengths:
        # get single prop + exclude label=1 (this labels words outside arguments)
        gold_prop = gold[start_p:start_p + l]
        pred_prop = pred[start_p:start_p + l]
        start_p += l
        gold_args = set(gold_prop)
        pred_args = set(pred_prop)
        if exclude_label_one:
            try:
                gold_args.remove(1)
                pred_args.remove(1)
            except KeyError:
                pass
        # hits + misses
        for arg in pred_args:  # assumes max number of args with same type in prop must be 1
            gold_span = np.where(gold_prop == arg)
            pred_span = np.where(pred_prop == arg)
            if np.array_equal(gold_span, pred_span):  # arg span can be broken into multiple phrases
                hits += 1
                gold_args.remove(arg)
            else:
                over_predictions += 1
        # any leftover predicted args are misses
        misses += len(gold_args)
    print('hits={:,}, over-predictions={:,}, misses={:,}'.format(hits, over_predictions, misses))
    # f1
    precision = hits / (hits + over_predictions)
    recall = hits / (hits + misses)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1