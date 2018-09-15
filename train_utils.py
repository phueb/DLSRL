import numpy as np
from pycm import ConfusionMatrix
from sklearn.metrics import f1_score, precision_score, recall_score


def shuffle_stack_pad(data, batch_size):
    """
    :param data: [list of word_id seqs, list of pred_id seqs, list of label_id seqs]
    :return: zero-padded matrices for each list of shape [num_seqs, max_seq_len]
    """
    shape0 = len(data[1])
    shape0_adj = shape0 - (shape0 % batch_size)
    shape1 = np.max([len(i) for i in data[1]])
    mats = [np.zeros((shape0_adj, shape1)).astype(np.int32) for _ in range(3)]
    rand_ids = np.random.choice(shape0, shape0_adj, replace=False)
    for sequences, mat in zip(data, mats):
        for n, rand_id in enumerate(rand_ids):
            seq = sequences[rand_id]
            mat[n, :len(seq)] = seq
    return mats


def count_zeros_from_end(s):
    res = 0
    it = iter(s[::-1])
    while next(it) == 0:
        res += 1
    else:
        return res


def get_batches(model, data, batch_size):
    x1, x2, y = shuffle_stack_pad(data, batch_size)
    num_batches = len(x1) // batch_size
    print('Generating {} batches with size {}'.format(num_batches, batch_size))
    for x1_b, x2_b, y_b in zip(np.vsplit(x1, num_batches),
                               np.vsplit(x2, num_batches),
                               np.vsplit(y, num_batches)):
        lengths = [len(s) - count_zeros_from_end(s) for s in x1_b]
        feed_dict = {model.word_ids: x1_b,
                     model.predicate_ids: x2_b,
                     model.label_ids: y_b,
                     model.lengths: lengths}
        yield feed_dict


def evaluate(config, model, sess, data):
    # predict
    batch_predictions = []
    batch_actuals = []
    for feed_dict in get_batches(model, data, config.train_batch_size):
        batch_pred = sess.run(model.predictions, feed_dict=feed_dict)
        batch_predictions.append(batch_pred.flatten())
        batch_actuals.append(feed_dict[model.label_ids].flatten())
    # confusion matrix based metrics
    a = np.concatenate(batch_actuals, axis=0)
    p = np.concatenate(batch_predictions, axis=0)
    nonzero_ids = np.nonzero(a)
    actual = a[nonzero_ids]
    predicted = p[nonzero_ids]
    cm = ConfusionMatrix(actual_vector=actual, predict_vector=predicted)
    # print
    print('/////////////////////////// EVALUATION')
    ppvs = np.array([i for i in cm.PPV.values() if i != 'None']).astype(np.float)
    tprs = np.array([i for i in cm.TPR.values() if i != 'None']).astype(np.float)
    f1s = np.array([i for i in cm.F1.values() if i != 'None']).astype(np.float)
    print('num total labels={} num nonzero labels={}'.format(len(a), len(actual)))
    print('PYMC    | precision={:.3f} recall={:.3f} f1={:.3f}'.format(
        np.mean(ppvs),
        np.mean(tprs),
        np.mean(f1s)))
    print('sklearn | precision={:.3f} recall={:.3f} f1={:.3f}'.format(
        f1_score(a, p, average='macro'),
        precision_score(a, p, average="macro"),
        recall_score(a, p, average="macro")))  # TODO how to get aggregate f1 score?