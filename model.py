import tensorflow as tf
from tensorflow.contrib.rnn import HighwayWrapper

# TODO implement:
# orthornormal initialization of weights
# gradient clipping
# variational dropout
# bidirectional


class Model():
    def __init__(self, config, data):
        self.batch_size = config.batch_size

        with tf.device('/gpu:0'):
            self.x = tf.placeholder(tf.int32, [None, None, None])  # TODO int?
            self.y = tf.placeholder(tf.int32, [None, None])
            # feed forward
            x_embedded = tf.gather(data.embeddings, self.x)  # TODO does gather work like embedding_lookup?
            all_states = []
            final_states = []
            for n in range(config.num_lstm_layers):
                print('Making layer {}...'.format(n))
                try:
                    prev_layer = all_states[-1]
                except IndexError:
                    prev_layer = x_embedded
                cell_input = prev_layer
                cell = HighwayWrapper(tf.contrib.rnn.LSTMCell(config.lstm_hidden_size))
                # calc state
                all_state, (c, h) = tf.nn.dynamic_rnn(cell, cell_input, dtype=tf.float32, scope=str(n))
                final_state = h
                # collect state
                all_states.append(all_state)
                final_states.append(final_state)
            # loss + step
            logits = all_states[-1]  # TODO
            labels = self.y  # TODO
            self.tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            self.step = tf.train.AdadeltaOptimizer(config.learning_rate).minimize(tf.reduce_mean(self.tf_loss))

