import time
import sys
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
from itertools import chain

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params as AllenParams
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.training import util as training_util
from allennlp.data.iterators import BucketIterator


from dlsrl.data import Data
from dlsrl.utils import get_batches, shuffle_stack_pad, count_zeros_from_end
from dlsrl.eval import print_f1, f1_official_conll05
from dlsrl.model import AllenSRLModel
from dlsrl import config


EVALUATE_ON_DEV = False  # TODO testing


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
    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # make local folder for saving checkpoint + events files
    local_job_p = config.LocalDirs.runs / params.job_name
    if not local_job_p.exists():
        local_job_p.mkdir(parents=True)

    # data
    data = Data(params)

    # vocab
    vocab = Vocabulary.from_instances(data.train_instances + data.dev_instances)

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
                "pretrained_file": str(config.Data.glove_path_local),
                "trainable": True
            }
        }
    })
    text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)

    # initializer
    initializer_params = [
        ("tag_projection_layer.*weight",
         AllenParams({ "type": "orthogonal"}))
    ]
    initializer = InitializerApplicator.from_params(initializer_params)

    # model
    model = AllenSRLModel(vocab=vocab,
                          text_field_embedder=text_field_embedder,
                          encoder=encoder,
                          initializer=initializer,
                          binary_feature_dim=100)
    model.cuda()  # TODO test

    from allennlp.common.checks import check_for_gpu
    check_for_gpu(device_id=0)

    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=params.learning_rate, eps=params.epsilon)

    bucket_batcher = BucketIterator(batch_size=params.batch_size,
                                    sorting_keys=[('tokens', "num_tokens")])

    bucket_batcher.index_with(vocab)

    # train loop
    dev_f1s = []
    train_start = time.time()
    for epoch in range(params.max_epochs):
        print()
        print('===========')
        print('Epoch: {}'.format(epoch))
        print('===========')

        # prepare data for epoch
        train_x1, train_x2, train_y = shuffle_stack_pad(data.train,
                                                        batch_size=params.batch_size)  # returns int32
        dev_x1, dev_x2, dev_y = shuffle_stack_pad(data.dev,
                                                  batch_size=config.Eval.dev_batch_size,
                                                  shuffle=False)

        # ----------------------------------------------- start evaluation

        if EVALUATE_ON_DEV:

            # per-label f1 evaluation
            all_gold_label_ids_no_pad = []
            all_pred_label_ids_no_pad = []

            # conll05 evaluation data
            all_sentence_pred_labels_no_pad = []
            all_sentence_gold_labels_no_pad = []
            all_verb_indices = []
            all_sentences_no_pad = []

            model.eval()
            for step, (x1_b, x2_b, y_b) in enumerate(get_batches(dev_x1, dev_x2, dev_y, config.Eval.dev_batch_size)):

                # get predicted label_ids from model

                softmax_2d = model()# TODO
                softmax_3d = np.reshape(softmax_2d, (*np.shape(x1_b), data.num_labels))  # 1st dim is batch_size
                batch_pred_label_ids = np.argmax(softmax_3d, axis=2)  # [batch_size, seq_length]
                batch_gold_label_ids = y_b  # [batch_size, seq_length]
                assert np.shape(batch_pred_label_ids) == (config.Eval.dev_batch_size, np.shape(x1_b)[1])

                # collect data for evaluation
                for x1_row, x2_row, gold_label_ids, pred_label_ids, in zip(x1_b,
                                                                           x2_b,
                                                                           batch_gold_label_ids,
                                                                           batch_pred_label_ids):

                    sentence_length = len(x1_row) - count_zeros_from_end(x1_row)

                    assert count_zeros_from_end(x1_row) == count_zeros_from_end(gold_label_ids)
                    sentence_gold_labels = [data.sorted_labels[i] for i in gold_label_ids]
                    sentence_pred_labels = [data.sorted_labels[i] for i in pred_label_ids]
                    verb_index = np.argmax(x2_row)
                    sentence = [data.sorted_words[i] for i in x1_row]

                    # collect data for conll-05 evaluation + remove padding
                    all_sentence_pred_labels_no_pad.append(sentence_pred_labels[:sentence_length])
                    all_sentence_gold_labels_no_pad.append(sentence_gold_labels[:sentence_length])
                    all_verb_indices.append(verb_index)
                    all_sentences_no_pad.append(sentence[:sentence_length])

                    # collect data for per-label evaluation
                    all_gold_label_ids_no_pad += list(gold_label_ids[:sentence_length])
                    all_pred_label_ids_no_pad += list(pred_label_ids[:sentence_length])

            print('Number of sentences to evaluate: {}'.format(len(all_sentences_no_pad)))

            for label in all_sentence_gold_labels_no_pad:
                assert label != config.Data.pad_label

            # evaluate f1 score computed over single labels (not spans)
            # f1_score expects 1D label ids (e.g. gold=[0, 2, 1, 0], pred=[0, 1, 1, 0])
            print_f1(epoch, 'weight', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='weighted'))
            print_f1(epoch, 'macro ', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='macro'))
            print_f1(epoch, 'micro ', f1_score(all_gold_label_ids_no_pad, all_pred_label_ids_no_pad, average='micro'))

            # evaluate with official conll05 perl script with Python interface provided by Allen AI NLP toolkit
            sys.stdout.flush()
            print('=============================================')
            print('Official Conll-05 Evaluation on Dev Split')
            dev_f1 = f1_official_conll05(all_sentence_pred_labels_no_pad,  # List[List[str]]
                                         all_sentence_gold_labels_no_pad,  # List[List[str]]
                                         all_verb_indices,  # List[Optional[int]]
                                         all_sentences_no_pad)  # List[List[str]]
            print_f1(epoch, 'conll-05', dev_f1)
            dev_f1s.append(dev_f1)
            print('=============================================')
            sys.stdout.flush()

        # ----------------------------------------------- end evaluation

        # train on batches
        model.train()
        train_generator = bucket_batcher(data.train_instances, num_epochs=1)
        for step, batch in enumerate(train_generator):

            # to cuda
            batch['tokens']['tokens'] = batch['tokens']['tokens'].cuda()
            batch['verb_indicator'] = batch['verb_indicator'].cuda()
            batch['tags'] = batch['tags'].cuda()

            a = batch['tokens']['tokens']
            b = batch['verb_indicator']
            c = batch['tags']

            print(a)
            print(b)
            print(c)

            assert a.is_cuda
            assert b.is_cuda
            assert c.is_cuda


            optimizer.zero_grad()

            output_dict = model(**batch)  # input is dict[str, torch tensor]
            loss = output_dict["loss"] + model.get_regularization_penalty()
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()
            training_util.rescale_gradients(model, params.max_grad_norm)
            optimizer.step()

            print(step, loss.item())  # TODO debugging

            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss.item(), (time.time() - train_start) // 60))


    # to pandas
    eval_epochs = np.arange(params.max_epochs)
    df_dev_f1 = pd.DataFrame(dev_f1s, index=eval_epochs, columns=['dev_f1'])
    df_dev_f1.name = 'dev_f1'

    return [df_dev_f1]
