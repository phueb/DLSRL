from typing import Optional, Dict, List, Any

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, \
    get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.util import rescale_gradients
from overrides import overrides
from torch.nn import Linear, Dropout, functional as F


class Model2(Model):
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
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None) -> None:
        super(Model2, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.encoder = encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(num_embeddings=2, embedding_dim=binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing

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
        rescale_gradients(self, self.max_grad_norm)
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