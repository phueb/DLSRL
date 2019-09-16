import tensorflow as tf

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

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                   embeddings_initializer=tf.keras.initializers.constant(embeddings),
                                                   mask_zero=True)  # TODO test mask_zero

        # TODO stack 8 LSTMs + residual connections

        self.lstm1 = tf.keras.layers.LSTM(params.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal',  # TODO orthonormal?
                                          dropout=0.0,
                                          recurrent_dropout=0.0,  # TODO recurrent dropout?
                                          return_sequences=True,
                                          go_backwards=False)
        self.lstm2 = tf.keras.layers.LSTM(params.cell_size,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          use_bias=True,
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal',  # TODO orthonormal?
                                          dropout=1 - params.keep_prob,
                                          recurrent_dropout=0.0,  # using recurrent dropout doesn't work
                                          return_sequences=True,
                                          go_backwards=False)  # TODO implement bi-directional

        self.dense_output = tf.keras.layers.Dense(num_labels,
                                                  input_shape=(None, params.cell_size),
                                                  activation='softmax')  # TODO test softmax

    def call(self, word_ids, predicate_ids, training):
        embedded = self.embedding(word_ids)
        # returns [batch_size, max_seq_len, embed_size]
        to_concat = tf.expand_dims(tf.cast(predicate_ids, tf.float32), -1)
        concatenated = tf.concat([embedded, to_concat], axis=2)
        # returns [batch_size, max_seq_len, embed_size + 1]

        mask = self.embedding.compute_mask(word_ids)
        encoded1 = self.lstm1(concatenated, mask=mask, training=training)
        # returns [batch_size, max_seq_len, cell_size]
        encoded2 = self.lstm2(encoded1, mask=mask, training=training)  # TODO backwards mask?
        # returns [batch_size, max_seq_len, cell_size]

        num_words = tf.size(word_ids)
        encoded_2d = tf.reshape(encoded2, [num_words, self.params.cell_size])
        # returns [num_words_in_batch, cell_size]
        logits = self.dense_output(encoded_2d)
        # returns [num_words_in_batch, num_labels]

        return logits

    # ------------------------------------------------------- unrelated to training

    def write_summaries(self):
        # TODO https://www.tensorflow.org/tensorboard/r2/get_started

        raise NotImplementedError