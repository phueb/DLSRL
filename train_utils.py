import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def shuffle_stack_pad(data, batch_size):
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
    rand_ids = np.random.choice(shape0, shape0_adj, replace=False)
    for sequences, mat in zip(data, mats):
        for n, rand_id in enumerate(rand_ids):
            seq = sequences[rand_id]
            mat[n, :len(seq)] = seq
    return mats  # x1, x2, y  # TODO data is intact here


def count_zeros_from_end(s):
    res = 0
    it = iter(s[::-1])
    while next(it) == 0:
        res += 1
    else:
        return res


def get_feed_dicts(x1, x2, y, model, batch_size, keep_prob):
    num_batches = len(x1) // batch_size
    print('Generating {} batches with size {}'.format(num_batches, batch_size))
    for x1_b, x2_b, y_b in zip(np.vsplit(x1, num_batches),
                               np.vsplit(x2, num_batches),
                               np.vsplit(y, num_batches)):
        lengths = [len(s) - count_zeros_from_end(s) for s in x1_b]
        max_batch_len = np.max(lengths)
        feed_dict = {model.word_ids: x1_b[:, :max_batch_len],
                     model.predicate_ids: x2_b[:, :max_batch_len],
                     model.label_ids: y_b[:, :max_batch_len],
                     model.keep_prob: keep_prob,
                     model.lengths: lengths}
        yield feed_dict


def remove_padding_and_flatten(mat, lengths):
    result = []
    for row, length in zip(mat, lengths):
        included = row[:length]
        result += included.tolist()
    return result


def evaluate(data, model, sess, epoch, global_step, word_dict, label_dict, batch_size=None):
    # make dev data
    batch_predictions = []
    batch_actuals = []
    if batch_size is None:
        batch_size = len(data[0])
        assert batch_size < 4096  # keep memory low when evaluating complete data (no batching)
    x1, x2, y = shuffle_stack_pad(data, batch_size)
    # eval dev data
    for feed_dict in get_feed_dicts(x1, x2, y, model, batch_size, keep_prob=1.0):
        # export confusion matrix
        summary = sess.run(model.merged2, feed_dict=feed_dict)
        model.train_writer.add_summary(summary, global_step)
        # get predictions
        batch_pred = sess.run(model.predictions, feed_dict=feed_dict)
        batch_act = feed_dict[model.label_ids]
        batch_wid = feed_dict[model.word_ids]
        lengths = feed_dict[model.lengths]
        batch_actuals.append(remove_padding_and_flatten(batch_act, lengths))
        batch_predictions.append(remove_padding_and_flatten(batch_pred, lengths))

        # TODO should words in sentence labeled as "O" receive error information? - right now they don't
        # print([(word_dict.idx2str[w_id], label_dict.idx2str[l_id])
        #        for w_id, l_id in zip(remove_padding_and_flatten(batch_wid, lengths)[:100],
        #                              remove_padding_and_flatten(batch_act, lengths)[:100])])

    # calc f1 score
    a = np.concatenate(batch_actuals, axis=0)
    p = np.concatenate(batch_predictions, axis=0)
    nonzero_ids = np.nonzero(a)
    a_no0 = a[nonzero_ids]
    p_no0 = p[nonzero_ids]

    # what is model predicting?
    for i, j in zip(a[:100], p[:100]):
        print('a="{}", p="{}"'.format(i, j))

    # print
    print('/////////////////////////// f1 EVALUATION START')
    print('num no-pad labels={:,} num non-zero labels={:,}'.format(len(a), len(a_no0)))
    print_f1(epoch, 'no-pad  ', 'macro', a, p)
    print_f1(epoch, 'no-pad  ', 'micro', a, p)
    print_f1(epoch, 'no-zero ', 'macro', a_no0, p_no0)
    print_f1(epoch, 'no-zero ', 'micro', a_no0, p_no0)
    print('/////////////////////////// f1 EVALUATION END ')


def print_f1(epoch, labels, method, a, p):
    print('epoch {:>3} labels={} method={} | p={:.2f} r={:.2f} f1={:.2f}'.format(
        epoch,
        labels,
        method,
        precision_score(a, p, average=method),
        recall_score(a, p, average=method),
        f1_score(a, p, average=method)))  # TODO which is conll script using?