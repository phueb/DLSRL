import numpy as np


def print_f1(epoch, method, f1):
    print('epoch {:>3} method={} | f1={:.2f}'.format(epoch, method, f1))


def f1_conll05(gold_label_ids, pred_label_ids, lengths):
    """

    compute res for arguments (spans of the same label) rather than single labels.
    an effort was made to compute the res score as closely as possible as it is done in the conll05 script.
    when using BIO tags, complete arguments are not compared (only B-to-B and I-to_I spans are compared)

    :param gold_label_ids: int32, [num_words]
    :param pred_label_ids: int32, [num_words]
    :param lengths: list of integers representing the length of sentences
    :return:
    """

    # number of labels must add up to total sum of sentence lengths (checks if padding has been removed)
    assert sum(lengths) == len(gold_label_ids)
    assert sum(lengths) == len(pred_label_ids)

    hits = 1
    over_predictions = 1
    misses = 1
    start_p = 0
    for l in lengths:

        # get single sentence (by getting all elements up to a certain length corresponding to a sentence)
        gold_prop = gold_label_ids[start_p:start_p + l]
        pred_prop = pred_label_ids[start_p:start_p + l]
        start_p += l
        gold_args = set(gold_prop)
        pred_args = set(pred_prop)

        # hits + misses + over_predictions
        for arg in pred_args:  # assumes max number of args with same type in prop must be 1
            gold_span = np.where(gold_prop == arg)
            pred_span = np.where(pred_prop == arg)
            if np.array_equal(gold_span, pred_span):  # arg span can be broken into multiple phrases
                hits += 1
                gold_args.remove(arg)
            else:
                over_predictions += 1
        misses += len(gold_args) # any leftover predicted args are misses

    print('hits={:,}, over-predictions={:,}, misses={:,}'.format(hits, over_predictions, misses))

    # f1
    precision = hits / (hits + over_predictions)
    recall = hits / (hits + misses)
    res = (2 * precision * recall) / (precision + recall)
    return res