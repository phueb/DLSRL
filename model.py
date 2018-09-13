import tensorflow as tf
from tensorflow.contrib.rnn import HighwayWrapper

# TODO implement:
# orthornormal initialization of weights
# gradient clipping
# variational dropout
# bidirectional
# highway wrapper - fix input output dimension incompatibility
# variable sized batches - use mask?


class Model():
    def __init__(self, config, embeddings, num_labels):
        self.batch_size = config.batch_size

        # embedding
        with tf.device('/cpu:0'):
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            x_embedded = tf.nn.embedding_lookup(embeddings, self.word_ids)

        # lstm
        with tf.device('/gpu:0'):
            all_outputs = []
            for n in range(config.num_lstm_layers):
                print('Making layer {}...'.format(n))
                try:
                    cell_input = all_outputs[-1]
                except IndexError:
                    self.predicate_ids = tf.placeholder(tf.float32, [None, None])
                    cell_input = tf.concat([x_embedded, tf.expand_dims(self.predicate_ids, -1)], axis=2)
                # cell = HighwayWrapper(tf.contrib.rnn.LSTMCell(config.lstm_hidden_size))
                cell = tf.contrib.rnn.LSTMCell(config.lstm_hidden_size)
                outputs, _ = tf.nn.dynamic_rnn(cell, cell_input, dtype=tf.float32, scope=str(n))
                all_outputs.append(outputs)

            # projection
            final_outputs = all_outputs[-1]  # [batch_size, max_seq_len, cell_size]
            shape0 = config.batch_size * config.max_seq_length
            final_outputs_2d = tf.reshape(final_outputs, [shape0, config.lstm_hidden_size])  # need [shape0, cell_size]
            wy = tf.get_variable('Wy', [config.lstm_hidden_size, num_labels])
            by = tf.get_variable('by', [num_labels])
            self.logits = tf.matmul(final_outputs_2d, wy) + by  # need [shape0, num_labels]

            # update
            self.label_ids = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_seq_len]
            self.labels = tf.reshape(self.label_ids, [-1])  # need [shape0]
            self.mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                           labels=self.labels))
            self.update = tf.train.AdadeltaOptimizer(
                learning_rate=config.learning_rate, epsilon=config.epsilon).minimize(tf.reduce_mean(self.mean_loss))