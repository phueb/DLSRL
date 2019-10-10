import numpy as np

from dlsrl.scorer_utils import convert_bio_tags_to_conll_format
from dlsrl.scorer import SrlEvalScorer


def evaluate_model_on_f1(model, params, vocab, bucket_batcher, instances):

    # inits
    all_bio_pred_labels = []  # no padding allowed
    all_bio_gold_labels = []  # no padding allowed
    all_verb_indices = []
    all_sentences = []

    model.eval()
    instances_generator = bucket_batcher(instances, num_epochs=1)
    for step, batch in enumerate(instances_generator):

        if len(batch['tags']) != params.batch_size:
            print('WARNING: Batch size is {}. Skipping'.format(len(batch['tags'])))
            continue

        # get predictions
        batch['training'] = False
        output_dict = model(**batch)            # input is dict[str, tensor]
        softmax_3d = output_dict['softmax_3d']  # [mb_size, max_sent_length, num_labels]

        # get words and verb_indices
        sentences_b = []
        verb_indices_b = []
        for row in batch['metadata']:
            sentence = row['words']
            verb_index = row['verb_index']
            sentences_b.append(sentence)
            verb_indices_b.append(verb_index)

        # do evaluation with numpy
        y_b = batch['tags'].cpu().numpy()
        x1_b = batch['tokens']['tokens'].cpu().numpy()
        x2_b = batch['verb_indicator'].cpu().numpy()

        # get gold and predicted label IDs
        batch_pred_label_ids = np.argmax(softmax_3d, axis=2)  # [batch_size, seq_length]
        batch_gold_label_ids = y_b  # [batch_size, seq_length]
        assert np.shape(batch_pred_label_ids) == (params.batch_size, np.shape(x1_b)[1])
        assert np.shape(batch_gold_label_ids) == (params.batch_size, np.shape(x1_b)[1])

        # collect data for evaluation
        for x1_row, x2_row, gold_label_ids, pred_label_ids, s, vi in zip(x1_b,
                                                                         x2_b,
                                                                         batch_gold_label_ids,
                                                                         batch_pred_label_ids,
                                                                         sentences_b,
                                                                         verb_indices_b):

            # convert IDs to tokens
            sentence_pred_labels = [vocab.get_token_from_index(i, namespace="labels")
                                    for i in pred_label_ids]
            sentence_gold_labels = [vocab.get_token_from_index(i, namespace="labels")
                                    for i in gold_label_ids]

            # collect data + remove padding
            sentence_length = len(s)
            all_bio_pred_labels.append(sentence_pred_labels[:sentence_length])
            all_bio_gold_labels.append(sentence_gold_labels[:sentence_length])
            all_verb_indices.append(vi)
            all_sentences.append(s)

    # convert to conll format
    all_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                tags in all_bio_pred_labels]
    all_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                           tags in all_bio_gold_labels]

    # SrlEvalScorer is taken from AllenAI NLP toolkit.
    # ignore_classes does not affect perl script, but affects f1 computed by Allen AI NLP toolkit
    span_metric = SrlEvalScorer(ignore_classes=["V"])
    span_metric(all_verb_indices,               # List[Optional[int]]
                all_sentences,                  # List[List[str]]
                all_conll_predicted_tags,       # List[List[str]]
                all_conll_gold_tags)            # List[List[str]]

    all_metrics = span_metric.get_metric()  # f1 is computed by Allen AI NLP toolkit given tp, fp, fn by perl script
    f1 = all_metrics["f1-measure-overall"]
    return f1
