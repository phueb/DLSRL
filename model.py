import tensorflow as tf
from tensorflow.contrib.rnn import HighwayWrapper

# TODO implement:
# orthornormal initialization of weights
# gradient clipping
# variational dropout
# bidirectional
# highway wrapper - fix input output dimension incompatibility


class Model():
    def __init__(self, config, data):
        self.batch_size = config.batch_size

        # embedding
        with tf.device('/cpu:0'):
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.feature_ids = tf.placeholder(tf.int32, [None, None])
            x_embedded = tf.gather(data.embeddings, self.word_ids)  # TODO does gather work like embedding_lookup?

        # lstm
        with tf.device('/gpu:0'):
            all_outputs = []
            for n in range(config.num_lstm_layers):
                print('Making layer {}...'.format(n))
                try:
                    cell_input = all_outputs[-1]
                except IndexError:
                    cell_input = x_embedded
                # TODO highway wrapper requires input and output dim to match
                # cell = HighwayWrapper(tf.contrib.rnn.LSTMCell(config.lstm_hidden_size))
                cell = tf.contrib.rnn.LSTMCell(config.lstm_hidden_size)
                outputs, _ = tf.nn.dynamic_rnn(cell, cell_input, dtype=tf.float32, scope=str(n))
                all_outputs.append(outputs)
            logits = all_outputs[-1]  # TODO

            # update
            self.y = tf.placeholder(tf.int32, [None, None])
            labels = tf.reshape(self.y, [tf.shape(self.y)[0] * config.max_seq_length, 1])
            self.mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            self.update = tf.train.AdadeltaOptimizer(
                learning_rate=config.learning_rate,epsilon=config.epsilon).minimize(tf.reduce_mean(self.mean_loss))