from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
import torch


class Model1(tf.keras.Model):
    """

    an incomplete tensorflow-based implementation of the Deep SRL model (described in He et al., 2017)

    """

    def __init__(self, embeddings, params, vocab, **kwargs):
        super(Model1, self).__init__(name='deep_lstm', **kwargs)

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


def masked_sparse_categorical_crossentropy(y_true, y_pred, mask_value):
    mask_value = tf.Variable(mask_value)
    mask = tf.equal(y_true, mask_value)
    mask = 1 - tf.cast(mask, tf.float32)

    # multiply categorical_crossentropy with the mask  (no reduce_sum operation is performed)
    _loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return tf.math.reduce_sum(_loss) / tf.math.reduce_sum(mask)