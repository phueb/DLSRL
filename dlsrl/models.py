from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from allennlp.common import Params as AllenParams
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.models.srl_util import convert_bio_tags_to_conll_format
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.nn.util import viterbi_decode
from allennlp.training.util import rescale_gradients
from overrides import overrides
from tensorflow.keras import layers
from torch.nn import Linear, Dropout, functional as F

from dlsrl import config
from dlsrl.scorer import SrlEvalScorer


def masked_sparse_categorical_crossentropy(y_true, y_pred, mask_value):
    mask_value = tf.Variable(mask_value)
    mask = tf.equal(y_true, mask_value)
    mask = 1 - tf.cast(mask, tf.float32)

    # multiply categorical_crossentropy with the mask  (no reduce_sum operation is performed)
    _loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return tf.math.reduce_sum(_loss) / tf.math.reduce_sum(mask)


class TensorflowSRLModel(tf.keras.Model):
    """

    an incomplete tensorflow-based implementation of the Deep SRL model (described in He et al., 2017)

    """

    def __init__(self, embeddings, params, vocab, **kwargs):
        super(TensorflowSRLModel, self).__init__(name='deep_lstm', **kwargs)

        self.num_classes = vocab.get_vocab_size("labels")
        self.padding_id = vocab.get_token_index(vocab._padding_token)
        assert self.padding_id == 0

        self.params = params
        num_embeddings, embed_size = embeddings.shape
        assert num_embeddings == vocab.get_vocab_size('tokens')

        print('Initializing keras model with in size = {} and out size = {}'.format(
            num_embeddings, self.num_classes))

        # ---------------------- define feed-forward ops

        # embed
        self.embedding1 = layers.Embedding(num_embeddings, embed_size,
                                           mask_zero=True,
                                           embeddings_initializer=tf.keras.initializers.constant(embeddings))
        self.embedding2 = layers.Embedding(2, params.binary_feature_dim)  # this is how He et al. did it

        # encode
        self.lstms = [layers.LSTM(params.hidden_size,
                                  return_sequences=True,
                                  go_backwards=bool(i % 2))
                      for i in range(params.num_layers)]

        # output
        self.dense_output = layers.Dense(self.num_classes, activation='softmax')

        # TODO missing:
        #  * control gates
        #  * orthonormal init of all weight matrices in LSTM
        #  * recurrent dropout but this prevents using cudnn and therefore takes 10 times longer

    def __call__(self, *args, **kwargs):
        """
        to keep interface consistent between models.
        mimic torch syntax for a forward pass which looks like:
        output_dict = model(**batch)
        """
        return self.call(*args, **kwargs)

    def eval(self):
        """
        to keep interface consistent between models
        """
        pass

    def train(self):
        """
        to keep interface consistent between models
        """
        pass

    def call(self,  # type: ignore
             tokens: Dict[str, torch.LongTensor],
             verb_indicator: torch.LongTensor,
             training: bool = False,
             tags: torch.LongTensor = None,
             metadata: List[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        x2_b is an array where each row is a one hot vector,
         where the hot index is the position of the verb in the sequence.
        this means x2_b should be embedded, retrieving either a vector with a single 1 or single 0.
        """

        # convert torch tensor to numpy
        x1_b = tokens['tokens'].numpy()
        x2_b = verb_indicator.numpy()

        # embed
        embedded1 = self.embedding1(x1_b)
        embedded2 = self.embedding1(x2_b)  # returns one of two trainable vectors of length params.binary_feature_dim
        mask = self.embedding1.compute_mask(x1_b)

        # encode
        encoded_lm0 = tf.concat([embedded1, embedded2], axis=2)  # [batch_size, max_seq_len, embed_size + feature dim]
        encoded_lm1 = None  # encoding at layer minus 1
        for layer, lstm in enumerate(self.lstms):
            encoded_lm2 = encoded_lm1  # in current layer (loop), what was previously 1 layer back is now 2 layers back
            encoded_lm1 = encoded_lm0  # in current layer (loop), what was previously 0 layer back is now 1 layers back
            if encoded_lm2 is None or tf.shape(encoded_lm2)[2] != self.params.hidden_size:
                encoded_lm2 = 0  # in layer 1 and 2, there should be no contribution from previous layers
            encoded_lm0 = lstm(encoded_lm1 + encoded_lm2, mask=mask, training=training)

        # output projection
        encoded_2d = tf.reshape(encoded_lm0 + encoded_lm1, [-1, self.params.hidden_size])
        softmax_2d = self.dense_output(encoded_2d)  # [num_words_in_batch, num_labels]
        softmax_3d = np.reshape(softmax_2d, (*np.shape(x1_b), self.num_classes))  # 1st dim is batch_size

        output_dict = {'softmax_2d': softmax_2d,
                       'softmax_3d': softmax_3d}
        return output_dict

    def train_on_batch(self, batch, optimizer):

        # to numpy
        y_b = batch['tags'].numpy()

        # forward + loss
        with tf.GradientTape() as tape:
            output_dict = self(**batch)
            y_true = y_b.reshape([-1])  # a tag for each word
            y_pred = output_dict['softmax_2d']  # a probability distribution for each word
            loss = masked_sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred, mask_value=self.padding_id)

        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss


class AllenSRLModel(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implementation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence.

    Philip Huebner: Do not use Viterbi decoding at inference

    Specifically, the model expects and outputs IOB2-formatted tags, where the
    B- tag is used in the beginning of every chunk (i.e. all chunks start with the B- tag).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing : ``float``, optional (default = 0.0)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric: ``bool``, optional (default = False)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False) -> None:
        super(AllenSRLModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.span_metric = SrlEvalScorer(ignore_classes=["V"])

        self.encoder = encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(num_embeddings=2, embedding_dim=binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric

        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               encoder.get_input_dim(),
                               "text embedding dim + verb indicator embedding dim",
                               "encoder input dim")
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None,
                training: bool = False,  # added by ph to make function consistent with other model
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence and the verb to compute the
            frame for, under 'words' and 'verb' keys, respectively.
        training : added by ph to make function consistent with other model - does nothing

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        # added by ph
        tokens['tokens'] = tokens['tokens'].cuda()
        verb_indicator = verb_indicator.cuda()
        if tags is not None:
            tags = tags.cuda()

        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()

        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities, "mask": mask}
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask,
                                                      label_smoothing=self._label_smoothing)
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_verb_indices = [example_metadata["verb_index"] for example_metadata in metadata]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]

                # Get the BIO tags from decode()
                batch_bio_predicted_tags = self.decode(output_dict).pop("tags")
                batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                              tags in batch_bio_predicted_tags]
                batch_bio_gold_tags = [example_metadata["gold_tags"] for example_metadata in metadata]
                batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                                         tags in batch_bio_gold_tags]
                self.span_metric(batch_verb_indices,
                                 batch_sentences,
                                 batch_conll_predicted_tags,
                                 batch_conll_gold_tags)
            output_dict["loss"] = loss

        # added by ph
        output_dict['softmax_3d'] = class_probabilities.detach().cpu().numpy()
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ph: Do NOT perform Viterbi decoding - we are interested in learning dynamics, not best performance
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])
        start_transitions = torch.zeros(num_labels)

        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix,
                                                        allowed_start_transitions=start_transitions)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)

            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "overall" in x}

    def train_on_batch(self, batch, optimizer):
        """
        written by ph to keep interface between models consistent
        """
        # to cuda
        batch['tokens']['tokens'] = batch['tokens']['tokens'].cuda()
        batch['verb_indicator'] = batch['verb_indicator'].cuda()
        batch['tags'] = batch['tags'].cuda()

        # forward + loss
        optimizer.zero_grad()
        output_dict = self(**batch)  # input is dict[str, tensor]
        loss = output_dict["loss"] + self.get_regularization_penalty()
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        # backward + update
        loss.backward()
        rescale_gradients(self, self.params.max_grad_norm)
        optimizer.step()

        return loss

    @staticmethod
    def handle_metadata(metadata):
        """
        added by ph.
        moved below code from self.forward() here because it was not used there
        """
        words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
        return list(words), list(verbs)


def make_model_and_optimizer(params, vocab, glove_path):
    if params.my_implementation:
        model = make_tensorflow_model(params, vocab, glove_path)
        optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate,
                                           epsilon=params.epsilon,
                                           clipnorm=params.max_grad_norm)
    else:
        model = make_allen_model(params, vocab, glove_path)
        optimizer = torch.optim.Adadelta(params=model.parameters(),
                                         lr=params.learning_rate,
                                         eps=params.epsilon)

    return model, optimizer


def make_allen_model(params, vocab, glove_path):
    # parameters for original model are specified here:
    # https://github.com/allenai/allennlp/blob/master/training_config/semantic_role_labeler.jsonnet

    glove_size = 100

    # encoder
    encoder_params = AllenParams(
        {'type': 'alternating_lstm',
         'input_size': glove_size + params.binary_feature_dim,
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
    model = AllenSRLModel(vocab=vocab,
                          text_field_embedder=text_field_embedder,
                          encoder=encoder,
                          initializer=initializer,
                          binary_feature_dim=params.binary_feature_dim,
                          ignore_span_metric=config.Eval.ignore_span_metric)
    model.cuda()
    model.params = params
    return model


def make_tensorflow_model(params, vocab, glove_path):

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

    model = TensorflowSRLModel(embeddings, params, vocab)
    return model
