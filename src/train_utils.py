import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def shuffle_stack_pad(data, batch_size, shuffle=True):
    """
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


def make_feed_dict(x1, x2, y, model, keep_prob):
    lengths = [len(s) - count_zeros_from_end(s) for s in x1]
    max_batch_len = np.max(lengths)
    feed_dict = {model.word_ids: x1[:, :max_batch_len],
                 model.predicate_ids: x2[:, :max_batch_len],
                 model.label_ids: y[:, :max_batch_len],
                 model.keep_prob: keep_prob,
                 model.lengths: lengths}
    return feed_dict


def evaluate(data, model, sess, epoch, global_step):
    # make dev data
    batch_size = len(data[0])
    assert batch_size <= 4096  # careful with large data
    x1, x2, y = shuffle_stack_pad(data, batch_size=batch_size, shuffle=False)
    feed_dict = make_feed_dict(x1, x2, y, model, keep_prob=1.0)

    # export confusion matrix
    summary = sess.run(model.cm_summary, feed_dict=feed_dict)
    model.train_writer.add_summary(summary, global_step)

    # get predictions
    p, a = sess.run([model.nonzero_predicted_label_ids, model.nonzero_label_ids_flat], feed_dict=feed_dict)

    # what is model predicting?
    for i, j in zip(a[:100], p[:100]):
        print('a="{}", p="{}"'.format(i, j))

    # calc f1
    print('/////////////////////////// f1 EVALUATION START')
    print('num labels={:,}'.format(len(a)))
    print_f1(epoch, 'macro', a, p)
    print_f1(epoch, 'micro', a, p)
    print('/////////////////////////// f1 EVALUATION END ')


def print_f1(epoch, method, a, p):
    print('epoch {:>3} method={} | p={:.2f} r={:.2f} f1={:.2f}'.format(
        epoch,
        method,
        precision_score(a, p, average=method),
        recall_score(a, p, average=method),
        f1_score(a, p, average=method)))  # TODO which is conll script using?