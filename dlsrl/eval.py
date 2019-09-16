import numpy as np


def print_f1(epoch, method, f1):
    print('epoch {:>3} method={} | f1={:.2f}'.format(epoch, method, f1))


def f1_conll05(gold, pred, lengths, exclude_label_one=True):  # TODO update this function
    """

    :param gold: int32, [num_words]
    :param pred: float32, [batch_size, num_labels]
    :param lengths:
    :param exclude_label_one:
    :return:
    """

    # TODO due to BIO tagging complete arguments are not compared (only B-to-B and I-to_I)

    hits = 1
    over_predictions = 1
    misses = 1
    start_p = 0
    for l in lengths:
        # get single sentence (by getting all elements up to a certain length corresponding to a sentence)
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