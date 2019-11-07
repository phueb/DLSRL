import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from allennlp.common import Params as AllenParams
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator

from dlsrl.model1 import Model1
from dlsrl.model2 import Model2


def make_model1(params, vocab, glove_path):

    if params.glove:
        print('Loading word embeddings from {}'.format(glove_path))
        df = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
        w2embed = {key: val.values for key, val in df.T.items()}
        embedding_size = next(iter(w2embed.items()))[1].shape[0]
        print('Glove embedding size={}'.format(embedding_size))
        print('Num embeddings in GloVe file: {}'.format(len(w2embed)))
    else:
        print('WARNING: Not loading GloVe embeddings')
        w2embed = {}
        embedding_size = params.binary_feature_dim

    # get info from Allen NLP vocab
    num_words = vocab.get_vocab_size('tokens')
    w2id = vocab.get_token_to_index_vocabulary('tokens')

    # assign embeddings
    embeddings = np.zeros((num_words, embedding_size), dtype=np.float32)
    num_found = 0
    for w, row_id in w2id.items():
        try:
            word_embedding = w2embed[w]
        except KeyError:
            embeddings[row_id] = np.random.standard_normal(embedding_size)
        else:
            embeddings[row_id] = word_embedding
            num_found += 1

    print('Found {}/{} GloVe embeddings'.format(num_found, num_words))
    # if this number is extremely low, then it is likely that Glove txt file was only
    # partially copied to shared drive (copying should be performed in terminal, not in GUI)

    model = Model1(embeddings, params, vocab)
    return model


def make_model2(params, vocab, glove_path):
    # parameters for original model are specified here:
    # https://github.com/allenai/allennlp/blob/master/training_config/semantic_role_labeler.jsonnet

    # do not change - these values are used in original model
    glove_size = 100
    binary_feature_dim = 100
    max_grad_norm = 1.0

    # encoder
    encoder_params = AllenParams(
        {'type': 'alternating_lstm',
         'input_size': glove_size + binary_feature_dim,
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
                "embedding_dim": glove_size,  # must match glove dimension
                "pretrained_file": str(glove_path),
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
    model = Model2(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   initializer=initializer,
                   binary_feature_dim=binary_feature_dim)
    model.cuda()
    model.max_grad_norm = max_grad_norm
    return model


def make_model_and_optimizer(params, vocab, glove_path):
    if params.model == 1:
        model = make_model1(params, vocab, glove_path)
        optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                           epsilon=params.epsilon,
                                           clipnorm=params.max_grad_norm)
        num_params = 'not implemented'  # TODO
    elif params.model == 2:  # 9M parameters
        # pytorch implementation of He et al., 2017 from Allen NLP toolkit
        model = make_model2(params, vocab, glove_path)
        optimizer = torch.optim.Adadelta(params=model.parameters(),
                                         lr=params.learning_rate,
                                         eps=params.epsilon)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise AttributeError('Invalid arg to model. Must be in [1, 2, 3]')

    print('Number of model parameters: {:,}'.format(num_params), flush=True)
    return model, optimizer