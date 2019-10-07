from typing import Dict, List, Any, Optional
import numpy as np
import tensorflow as tf
import torch
from allennlp.common import Params as AllenParams
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.models.srl_util import convert_bio_tags_to_conll_format
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, \
    get_lengths_from_binary_sequence_mask, viterbi_decode
from overrides import overrides
from tensorflow.keras import layers
from torch.nn import Linear, Dropout, functional as F

from dlsrl import config
from dlsrl.scorer import SrlEvalScorer


class TensorflowSRLModel(tf.keras.Model):

    def __init__(self,  data, params, vocab, **kwargs):
        super(TensorflowSRLModel, self).__init__(name='deep_lstm', **kwargs)

        self.num_classes = vocab.get_vocab_size("labels")

        print('Initializing keras model with in size = {} and out size = {}'.format(
            len(data.embeddings), self.num_classes))

        self.params = params
        vocab_size, embed_size = data.embeddings.shape

        # embed word_ids
        self.embedding1 = layers.Embedding(vocab_size, embed_size,
                                           embeddings_initializer=tf.keras.initializers.constant(data.embeddings),
                                           mask_zero=True)  # TODO test mask_zero

        # embed predicate_ids
        self.embedding2 = layers.Embedding(2, 100)  # TODO test - this is how He et al. did it

        # TODO control gates + orthonormal init of all weight matrices in LSTM

        # He et al., 2017 used recurrent dropout but this prevents using cudnn and takes 10 times longer

        self.lstm1 = layers.LSTM(params.hidden_size, return_sequences=True)
        self.lstm2 = layers.LSTM(params.hidden_size, return_sequences=True,
                                 go_backwards=True)
        self.lstm3 = layers.LSTM(params.hidden_size, return_sequences=True)
        self.lstm4 = layers.LSTM(params.hidden_size, return_sequences=True,
                                 go_backwards=True)
        # self.lstm5 = layers.LSTM(params.hidden_size, return_sequences=True)
        # self.lstm6 = layers.LSTM(params.hidden_size, return_sequences=True,
        #                          go_backwards=True)
        # self.lstm7 = layers.LSTM(params.hidden_size, return_sequences=True)
        # self.lstm8 = layers.LSTM(params.hidden_size, return_sequences=True,
        #                          go_backwards=True)

        self.dense_output = layers.Dense(self.num_classes, activation='softmax')

    def __call__(self, *args, **kwargs):
        """
        to keep interface consistent with torch:
        output_dict = model(**batch)
        """
        return self.call(*args, **kwargs)

    def eval(self):
        """
        to keep interface consistent with torch:
        """
        pass

    def train(self):
        """
        to keep interface consistent with torch
        """
        pass

    def call(self,  # type: ignore
             tokens: Dict[str, np.ndarray],
             verb_indicator: np.ndarray,
             training: bool,
             tags: np.ndarray = None,
             metadata: List[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        x2_b is an array where each row is a one hot vector,
         where the hot index is the position of the verb in the sequence.
        this means x2_b should be embedded, retrieving either a vector with a single 1 or single 0.
        """

        x1_b = tokens['tokens']
        x2_b = verb_indicator

        # embed
        embedded1 = self.embedding1(x1_b)
        embedded2 = self.embedding1(x2_b)  # returns one of two trainable vectors of length 100

        # encoding
        encoded0 = tf.concat([embedded1, embedded2], axis=2)
        # returns [batch_size, max_seq_len, embed_size + 100]

        mask = self.embedding1.compute_mask(x1_b)
        encoded1 = self.lstm1(encoded0, mask=mask, training=training)
        encoded2 = self.lstm2(encoded1, mask=mask, training=training)
        encoded3 = self.lstm3(encoded1 + encoded2, mask=mask, training=training)
        encoded4 = self.lstm4(encoded2 + encoded3, mask=mask, training=training)
        # encoded5 = self.lstm5(encoded3 + encoded4, mask=mask, training=training)
        # encoded6 = self.lstm6(encoded4 + encoded5, mask=mask, training=training)
        # encoded7 = self.lstm7(encoded5 + encoded6, mask=mask, training=training)
        # encoded8 = self.lstm8(encoded6 + encoded7, mask=mask, training=training)
        # returns [batch_size, max_seq_len, hidden_size]

        encoded_2d = tf.reshape(encoded3 + encoded4, [-1, self.params.hidden_size])
        softmax_2d = self.dense_output(encoded_2d)
        # returns [num_words_in_batch, num_labels]

        output_dict = {'softmax_2d': softmax_2d}
        return output_dict


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
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
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

        words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
        if metadata is not None:
            output_dict["words"] = list(words)
            output_dict["verb"] = list(verbs)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Do NOT perform Viterbi decoding.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []

        # transition matrices contain only ones (and no -inf, which would signal illegal transition)
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


def make_model(data, params, vocab):
    if params.my_implementation:
        return TensorflowSRLModel(data, params, vocab)
    else:
        return make_allen_model(data, params, vocab)


def make_allen_model(data, params, vocab):
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