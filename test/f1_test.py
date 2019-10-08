import unittest

from dlsrl.scorer import SrlEvalScorer
from dlsrl.scorer_utils import convert_bio_tags_to_conll_format


class MyTest(unittest.TestCase):

    def test_f1(self):
        """
        test that f1 evaluation works
        """
        sentence = "The economy 's temperature will be taken from several vantage points this week , with readings on trade , output , housing and inflation .".split()
        bio_pred_tag_seq = 'B-A1 I-A1 I-A1 I-A1 B-AM-MOD O B-V B-A2 I-A2 I-A2 I-A2 B-AM-TMP I-AM-TMP O B-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV O'.split()
        bio_gold_tags = 'B-A1 I-A1 I-A1 I-A1 B-AM-MOD O B-V B-A2 I-A2 I-A2 I-A2 B-AM-TMP I-AM-TMP O B-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV O'.split()

        all_bio_pred_labels = [bio_pred_tag_seq]
        all_bio_gold_labels = [bio_gold_tags]
        all_sentences = [sentence]
        all_verb_indices = [6]

        # convert to conll format
        all_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                    tags in all_bio_pred_labels]
        all_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                               tags in all_bio_gold_labels]

        # SrlEvalScorer is taken from AllenAI NLP toolkit.
        # ignore_classes does not affect perl script, but affects f1 computed by Allen AI NLP toolkit
        span_metric = SrlEvalScorer(ignore_classes=["V"])
        span_metric(all_verb_indices,  # List[Optional[int]]
                    all_sentences,  # List[List[str]]
                    all_conll_predicted_tags,  # List[List[str]]
                    all_conll_gold_tags)  # List[List[str]]

        all_metrics = span_metric.get_metric()  # f1 is computed by Allen AI NLP toolkit given tp, fp, fn by perl script
        f1 = all_metrics["f1-measure-overall"]
        return f1 > 0.99


if __name__ == '__main__':
    unittest.main()