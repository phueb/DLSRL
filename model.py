import tensorflow as tf
from tensorflow.contrib.rnn import HighwayWrapper, LSTMCell, DropoutWrapper

PARALLEL_ITERATIONS = 32

# TODO visualize graph on tensorboard


def orthonorm(shape, dtype=tf.float32,  # TODO only works for square (recurrent) weights
              partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u


def bidirectional_lstm(inputs, num_layers, size, keep_prob, lengths):
    outputs = inputs
    for layer in range(num_layers // 2):
        print('Layer {}: Creating forward + backward LSTM'.format(layer))
        with tf.variable_scope('bilstm_{}'.format(layer), reuse=tf.AUTO_REUSE):
            # highway wrapper implements transform gates - weighted addition of input to output
            cell_fw = HighwayWrapper(DropoutWrapper(LSTMCell(size),
                                                    state_keep_prob=keep_prob))
            cell_bw = HighwayWrapper(DropoutWrapper(LSTMCell(size),
                                                    state_keep_prob=keep_prob))
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                             cell_bw,
                                                                             outputs,
                                                                             sequence_length=lengths,
                                                                             dtype=tf.float32)
            outputs = tf.add(output_fw, output_bw)  # result is [B, T, cell_size]

    return outputs


class Model():
    def __init__(self, config, embeddings, num_labels):

        # embedding
        with tf.device('/cpu:0'):
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            x_embedded = tf.nn.embedding_lookup(embeddings, self.word_ids)

        # stacked bilstm
        with tf.device('/gpu:0'):
            self.predicate_ids = tf.placeholder(tf.float32, [None, None])
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.lengths = tf.placeholder(tf.int32, [None])
            inputs = tf.concat([x_embedded, tf.expand_dims(self.predicate_ids, -1)], axis=2)
            final_outputs = bidirectional_lstm(inputs, config.num_layers, config.cell_size, self.keep_prob, self.lengths)

            # with tf.variable_scope('ortho_inits', initializer=orthonorm):
            #     self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            #     fw_cells = [DropoutWrapper(LSTMCell(config.cell_size),
            #                                variational_recurrent=True,
            #                                dtype=tf.float32,
            #                                input_keep_prob=1.0,
            #                                output_keep_prob=1.0,
            #                                state_keep_prob=self.keep_prob)
            #                 for _ in range(config.num_layers // 2)]
            #     bw_cells = [DropoutWrapper(LSTMCell(config.cell_size),  # TODO dropout or residuals first?
            #                                variational_recurrent=True,
            #                                dtype=tf.float32,
            #                                input_keep_prob=1.0,
            #                                output_keep_prob=1.0,
            #                                state_keep_prob=self.keep_prob)
            #                 for _ in range(config.num_layers // 2)]
            #
            #     self.lengths = tf.placeholder(tf.int32, [None])
            #     (final_outputs, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            #         cells_fw=fw_cells,
            #         cells_bw=bw_cells,
            #         inputs=inputs,
            #         sequence_length=self.lengths,  # TODO test
            #         dtype=tf.float32,
            #         parallel_iterations=PARALLEL_ITERATIONS)
            #     # final_outputs is depth-concatenated  # TODO add highway connections - but how when outputs are concatenated?

            # projection
            shape0 = tf.shape(final_outputs)[0] * tf.shape(final_outputs)[1]  # both batch_size and seq_len are dynamic
            final_outputs_2d = tf.reshape(final_outputs, [shape0, config.cell_size])  # need [shape0, cell_size]
            wy = tf.get_variable('Wy', [config.cell_size, num_labels])
            by = tf.get_variable('by', [num_labels])
            self.logits = tf.matmul(final_outputs_2d, wy) + by  # need [shape0, num_labels]

            # loss
            self.label_ids = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_seq_len]
            label_ids_flat = tf.reshape(self.label_ids, [-1])  # need [shape0]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=label_ids_flat)
            mask = tf.sign(tf.to_float(label_ids_flat))  # requires that there is no zero label (only zero padding)
            masked_losses = mask * losses
            self.mean_loss = tf.reduce_mean(masked_losses)  # TODO test

            # update
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.learning_rate, epsilon=config.epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.mean_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
            self.update = optimizer.apply_gradients(zip(gradients, variables))

            # for metrics
            self.predictions = tf.argmax(tf.nn.softmax(self.logits), axis=1)

