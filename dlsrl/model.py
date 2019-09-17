import tensorflow as tf
from tensorflow.keras import layers

from dlsrl import config


# TODO use fucntional API: define two models: 1 for embedding and another for LSTms
#  and then define a model that brings these 2 together
#  as in: https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models

class Model(tf.keras.Model):

    def __init__(self, params, embeddings, num_labels, name='deep_lstm', **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

        print('Initializing keras model with in size = {} and out size = {}'.format(
            len(embeddings), num_labels))

        self.params = params
        vocab_size, embed_size = embeddings.shape

        self.embedding = layers.Embedding(vocab_size, embed_size,
                                          embeddings_initializer=tf.keras.initializers.constant(embeddings),
                                          mask_zero=True)  # TODO test mask_zero

        # TODO residual connections + orthonormal init of all weight matrices in LSTM

        # The output of the Bidirectional RNN will be, by default,
        # the sum of the forward layer output and the backward layer output

        self.lstm1 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1-params.keep_prob)
        self.lstm2 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1-params.keep_prob,
                                 go_backwards=True)
        # self.lstm3 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1 - params.keep_prob)
        # self.lstm4 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1 - params.keep_prob,
        #                          go_backwards=True)
        # self.lstm5 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1 - params.keep_prob)
        # self.lstm6 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1 - params.keep_prob,
        #                          go_backwards=True)
        # self.lstm7 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1 - params.keep_prob)
        # self.lstm8 = layers.LSTM(params.cell_size, return_sequences=True, dropout=1 - params.keep_prob,
        #                          go_backwards=True)

        self.dense_output = layers.Dense(num_labels,
                                         activation='softmax')  # TODO test softmax

    def call(self, word_ids, predicate_ids, training):
        embedded = self.embedding(word_ids)
        # returns [batch_size, max_seq_len, embed_size]
        to_concat = tf.expand_dims(tf.cast(predicate_ids, tf.float32), -1)
        encoded0 = tf.concat([embedded, to_concat], axis=2)
        # returns [batch_size, max_seq_len, embed_size + 1]

        mask = self.embedding.compute_mask(word_ids)
        encoded1 = self.lstm1(encoded0, mask=mask, training=training)
        encoded2 = self.lstm2(encoded1, mask=mask, training=training)
        # encoded3 = self.lstm3(encoded1 + encoded2, mask=mask, training=training)
        # encoded4 = self.lstm4(encoded2 + encoded3, mask=mask, training=training)
        # encoded5 = self.lstm5(encoded3 + encoded4, mask=mask, training=training)
        # encoded6 = self.lstm6(encoded4 + encoded5, mask=mask, training=training)
        # encoded7 = self.lstm7(encoded5 + encoded6, mask=mask, training=training)
        # encoded8 = self.lstm8(encoded6 + encoded7, mask=mask, training=training)
        # returns [batch_size, max_seq_len, cell_size]

        num_words = tf.size(word_ids)
        encoded_2d = tf.reshape(encoded1 + encoded2, [num_words, self.params.cell_size])  # TODO 1 + 2
        # returns [num_words_in_batch, cell_size]
        logits = self.dense_output(encoded_2d)
        # returns [num_words_in_batch, num_labels]

        return logits

    # ------------------------------------------------------- unrelated to training

    def write_summaries(self):
        # TODO https://www.tensorflow.org/tensorboard/r2/get_started

        raise NotImplementedError