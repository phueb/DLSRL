import time
import sys
import numpy as np
import pandas as pd
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params as AllenParams
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.training import util as training_util
from allennlp.data.iterators import BucketIterator


from dlsrl.data import Data
from dlsrl.eval import f1_official_conll05
from dlsrl.model import AllenSRLModel
from dlsrl import config


class Params:

    def __init__(self, param2val):
        param2val = param2val.copy()

        self.param_name = param2val.pop('param_name')
        self.job_name = param2val.pop('job_name')

        self.param2val = param2val

    def __getattr__(self, name):
        if name in self.param2val:
            return self.param2val[name]
        else:
            raise AttributeError('No such attribute')

    def __str__(self):
        res = ''
        for k, v in sorted(self.param2val.items()):
            res += '{}={}\n'.format(k, v)
        return res


def main(param2val):

    if config.Global.local:
        print('WARNING: Loading data locally because config.Global.local=True')
        config.RemoteDirs = config.LocalDirs

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # make local folder for saving checkpoint + events files
    local_job_p = config.LocalDirs.runs / params.job_name
    if not local_job_p.exists():
        local_job_p.mkdir(parents=True)

    # data + vocab
    print('Building data...')
    data = Data(params)
    vocab = Vocabulary.from_instances(data.train_instances + data.dev_instances)

    # batching
    bucket_batcher = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(vocab)

    # model
    print('Building model...')
    model = make_model(data, params, vocab)

    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=params.learning_rate, eps=params.epsilon)

    # train loop
    dev_f1s = []
    train_start = time.time()
    for epoch in range(params.max_epochs):
        print()
        print('===========')
        print('Epoch: {}'.format(epoch))
        print('===========')

        # ----------------------------------------------- start evaluation

        # conll05 evaluation data
        all_sentence_pred_labels_no_pad = []
        all_sentence_gold_labels_no_pad = []
        all_verb_indices = []
        all_sentences_no_pad = []

        model.eval()
        dev_generator = bucket_batcher(data.dev_instances, num_epochs=1)
        for step, batch in enumerate(dev_generator):

            if len(batch['tags']) != params.batch_size:
                print('WARNING: Batch size is {}. Skipping'.format(len(batch['tags'])))
                continue

            y_b = batch['tags'].cpu().numpy()
            x1_b = batch['tokens']['tokens'].cpu().numpy()
            x2_b = batch['verb_indicator'].cpu().numpy()

            # to CUDA
            batch['tokens']['tokens'] = batch['tokens']['tokens'].cuda()
            batch['verb_indicator'] = batch['verb_indicator'].cuda()
            batch['tags'] = batch['tags'].cuda()

            # get predictions (softmax_3d has shape [mb_size, max_sent_length, num_labels])
            output_dict = model(**batch)  # input is dict[str, torch tensor]
            softmax_3d = output_dict['class_probabilities'].detach().cpu().numpy()  # 1st dim is batch_size

            # get words and verb_indices
            sentences_b = []
            verb_indices_b = []
            for row in batch['metadata']:  # this is correct
                sentence = row['words']
                verb_index = row['verb_index']
                sentences_b.append(sentence)
                verb_indices_b.append(verb_index)

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

                # collect data for conll-05 evaluation + remove padding
                sentence_length = len(s)
                all_sentence_pred_labels_no_pad.append(sentence_pred_labels[:sentence_length])
                all_sentence_gold_labels_no_pad.append(sentence_gold_labels[:sentence_length])
                all_verb_indices.append(vi)
                all_sentences_no_pad.append(s)

        for label in all_sentence_gold_labels_no_pad:
            assert label != config.Data.pad_label

        # evaluate with official conll05 perl script with Python interface provided by Allen AI NLP toolkit
        sys.stdout.flush()
        print('=============================================')
        print('Official Conll-05 Evaluation on Dev Split')
        dev_f1 = f1_official_conll05(all_sentence_pred_labels_no_pad,  # List[List[str]]
                                     all_sentence_gold_labels_no_pad,  # List[List[str]]
                                     all_verb_indices,  # List[Optional[int]]
                                     all_sentences_no_pad)  # List[List[str]]
        print('=============================================')
        sys.stdout.flush()

        # collect
        dev_f1s.append(dev_f1)

        # ----------------------------------------------- end evaluation

        # train on batches
        model.train()
        train_generator = bucket_batcher(data.train_instances, num_epochs=1)
        for step, batch in enumerate(train_generator):

            # to cuda
            batch['tokens']['tokens'] = batch['tokens']['tokens'].cuda()
            batch['verb_indicator'] = batch['verb_indicator'].cuda()
            batch['tags'] = batch['tags'].cuda()

            # forward + loss
            optimizer.zero_grad()
            output_dict = model(**batch)  # input is dict[str, torch tensor]
            loss = output_dict["loss"] + model.get_regularization_penalty()
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            # backward + update
            loss.backward()
            training_util.rescale_gradients(model, params.max_grad_norm)
            optimizer.step()

            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    # to pandas
    eval_epochs = np.arange(params.max_epochs)
    df_dev_f1 = pd.DataFrame(dev_f1s, index=eval_epochs, columns=['dev_f1'])
    df_dev_f1.name = 'dev_f1'

    return [df_dev_f1]


def make_model(data, params, vocab):
    # parameters for original model are specified here:
    # https://github.com/allenai/allennlp/blob/master/training_config/semantic_role_labeler.jsonnet

    # encoder
    encoder_params = AllenParams(
        {'type': 'alternating_lstm',
         'input_size': 200,  # this is glove size + binary feature embedding size = 200
         'hidden_size': params.hidden_size,
         'num_layers': params.num_layers,
         'use_highway': True,
         'recurrent_dropout_probability': 0.1})
    encoder = Seq2SeqEncoder.from_params(encoder_params)

    # embedder
    embedder_params = AllenParams({
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 100,  # must match glove dimension
                "pretrained_file": str(data.glove_path),
                "trainable": True
            }
        }
    })
    text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)

    # initializer
    initializer_params = [
        ("tag_projection_layer.*weight",
         AllenParams({"type": "orthogonal"}))
    ]
    initializer = InitializerApplicator.from_params(initializer_params)

    # model
    model = AllenSRLModel(vocab=vocab,
                          text_field_embedder=text_field_embedder,
                          encoder=encoder,
                          initializer=initializer,
                          binary_feature_dim=100,
                          ignore_span_metric=config.Eval.ignore_span_metric)  # TODO test
    model.cuda()

    from allennlp.common.checks import check_for_gpu
    check_for_gpu(device_id=0)

    return model