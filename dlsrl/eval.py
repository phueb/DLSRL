import numpy as np

from dlsrl.scorer_utils import convert_bio_tags_to_conll_format
from dlsrl.scorer import SrlEvalScorer


def f1_official_conll05(batch_bio_predicted_tags,   # List[List[str]]
                        batch_bio_gold_tags,        # List[List[str]]
                        batch_verb_indices,         # List[Optional[int]]
                        batch_sentences):           # List[List[str]]

    batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                  tags in batch_bio_predicted_tags]
    batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                             tags in batch_bio_gold_tags]

    span_metric = SrlEvalScorer(ignore_classes=["V"])  # SrlEvalScorer is available from AllenAI NLP toolkit
    span_metric(batch_verb_indices,             # List[Optional[int]]
                batch_sentences,                # List[List[str]]
                batch_conll_predicted_tags,     # List[List[str]]
                batch_conll_gold_tags)          # List[List[str]]

    all_metrics = span_metric.get_metric()
    for key in ["precision-overall",
                "recall-overall",
                "f1-measure-overall"]:
        print('{}={:.2f}'.format(key, all_metrics[key]))

    return all_metrics["f1-measure-overall"]


def print_f1(epoch, method, f1):
    print('epoch {:>3} method={} | f1={:.2f}'.format(epoch, method, f1))


def f1_naive(gold_label_ids, pred_label_ids, lengths):
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


