
from dlsrl.scorer_utils import convert_bio_tags_to_conll_format
from dlsrl.scorer import SrlEvalScorer
from dlsrl import config


def f1_official_conll05(batch_bio_predicted_tags,   # List[List[str]]
                        batch_bio_gold_tags,        # List[List[str]]
                        batch_verb_indices,         # List[Optional[int]]
                        batch_sentences,            # List[List[str]]
                        ):

    batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                  tags in batch_bio_predicted_tags]
    batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                             tags in batch_bio_gold_tags]

    # SrlEvalScorer is available from AllenAI NLP toolkit
    span_metric = SrlEvalScorer(ignore_classes=["V", config.Data.pad_label])
    span_metric(batch_verb_indices,             # List[Optional[int]]
                batch_sentences,                # List[List[str]]
                batch_conll_predicted_tags,     # List[List[str]]
                batch_conll_gold_tags)          # List[List[str]]

    all_metrics = span_metric.get_metric()  # f1 is computed by Allen AI NLP toolkit given tp, fp, fn by perl script

    return all_metrics["f1-measure-overall"]


def print_f1(epoch, method, f1):
    print('epoch {:>3} method={} | f1={:.2f}'.format(epoch, method, f1))





