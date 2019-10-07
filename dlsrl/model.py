from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np


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
        # init = np.vstack((np.zeros(embed_size),  # TODO test
        #                   np.ones(embed_size)))
        # print('Initializing binary feature embedding with:')
        # print(init)
        # self.embedding2 = layers.Embedding(2, embed_size,  # He et al., 2017 use 100 here too
        #                                    embeddings_initializer=tf.keras.initializers.constant(init)
        #                                    )

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

    def call(self, x1_b, x2_b, training):

        embedded1 = self.embedding1(x1_b)
        # embedded2 = self.embedding2(x2_b)
        # returns [batch_size, max_seq_len, embed_size]

        # TODO test one-hot static embedding of predicate feature
        predicate_feature = tf.expand_dims(tf.cast(x2_b, tf.float32), -1)
        encoded0 = tf.concat([embedded1, predicate_feature], axis=2)
        # returns [batch_size, max_seq_len, embed_size + 1]

        # encoded0 = tf.concat([embedded1, embedded2], axis=2)
        # returns [batch_size, max_seq_len, embed_size + embed_size]

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

    # ------------------------------------------------------- unrelated to training

    def write_summaries(self):
        # TODO https://www.tensorflow.org/tensorboard/r2/get_started

        raise NotImplementedError