import tensorflow as tf
from tensorflow.keras import layers

import numpy as np


class Model(tf.keras.Model):

    def __init__(self, params, embeddings, num_labels, name='deep_lstm', **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

        print('Initializing keras model with in size = {} and out size = {}'.format(
            len(embeddings), num_labels))

        self.params = params
        vocab_size, embed_size = embeddings.shape

        # embed word_ids
        self.embedding1 = layers.Embedding(vocab_size, embed_size,
                                           embeddings_initializer=tf.keras.initializers.constant(embeddings),
                                           mask_zero=True)  # TODO test mask_zero

        # embed predicate_ids
        init = np.vstack((np.zeros(embed_size),  # TODO test
                          np.ones(embed_size)))
        print('Initializing binary feature embedding with:')
        print(init)
        self.embedding2 = layers.Embedding(2, embed_size,  # He et al., 2017 use 100 here too
                                           embeddings_initializer=tf.keras.initializers.constant(init)
                                           )

        # TODO control gates + orthonormal init of all weight matrices in LSTM

        # He et al., 2017 used recurrent dropout but this prevents using cudnn and takes 10 times longer

        self.lstm1 = layers.LSTM(params.cell_size, return_sequences=True)
        self.lstm2 = layers.LSTM(params.cell_size, return_sequences=True,
                                 go_backwards=True)
        self.lstm3 = layers.LSTM(params.cell_size, return_sequences=True)
        self.lstm4 = layers.LSTM(params.cell_size, return_sequences=True,
                                 go_backwards=True)
        self.lstm5 = layers.LSTM(params.cell_size, return_sequences=True)
        self.lstm6 = layers.LSTM(params.cell_size, return_sequences=True,
                                 go_backwards=True)
        self.lstm7 = layers.LSTM(params.cell_size, return_sequences=True)
        self.lstm8 = layers.LSTM(params.cell_size, return_sequences=True,
                                 go_backwards=True)

        self.dense_output = layers.Dense(num_labels, activation='softmax')

    def call(self, word_ids, predicate_ids, training):
        embedded1 = self.embedding1(word_ids)
        embedded2 = self.embedding2(predicate_ids)
        # returns [batch_size, max_seq_len, embed_size]

        encoded0 = tf.concat([embedded1, embedded2], axis=2)
        # returns [batch_size, max_seq_len, embed_size + embed_size]

        mask = self.embedding1.compute_mask(word_ids)
        encoded1 = self.lstm1(encoded0, mask=mask, training=training)
        encoded2 = self.lstm2(encoded1, mask=mask, training=training)
        encoded3 = self.lstm3(encoded1 + encoded2, mask=mask, training=training)
        encoded4 = self.lstm4(encoded2 + encoded3, mask=mask, training=training)
        encoded5 = self.lstm5(encoded3 + encoded4, mask=mask, training=training)
        encoded6 = self.lstm6(encoded4 + encoded5, mask=mask, training=training)
        encoded7 = self.lstm7(encoded5 + encoded6, mask=mask, training=training)
        encoded8 = self.lstm8(encoded6 + encoded7, mask=mask, training=training)
        # returns [batch_size, max_seq_len, cell_size]

        encoded_2d = tf.reshape(encoded7 + encoded8, [-1, self.params.cell_size])
        softmax_2d = self.dense_output(encoded_2d)
        # returns [num_words_in_batch, num_labels]

        return softmax_2d

    # ------------------------------------------------------- unrelated to training

    def write_summaries(self):
        # TODO https://www.tensorflow.org/tensorboard/r2/get_started

        raise NotImplementedError