import tensorflow as tf
from tensorflow.contrib.rnn import HighwayWrapper, LSTMCell, DropoutWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

PARALLEL_ITERATIONS = 32

# TODO visualize graph on tensorboard


def orthonorm(shape, dtype=tf.float32,  # TODO only works for square (recurrent) weights
              partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u


def _reverse(input_, seq_lengths, seq_dim, batch_dim):  # reverses sequences with right-padding correctly
    return array_ops.reverse_sequence(
        input=input_, seq_lengths=seq_lengths,
        seq_dim=seq_dim, batch_dim=batch_dim)


def bidirectional_lstms_stacked(inputs, num_layers, size, keep_prob, lengths):
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


def bidirectional_lstms_interleaved(inputs, num_layers, size, keep_prob, lengths):
    outputs = inputs
    for layer in range(num_layers):
        print('Layer {}: Creating {} LSTM'.format(layer, 'backw.' if layer % 2 else 'forw.'))  # backwards if layer odd
        with tf.variable_scope('bilstm_{}'.format(layer), reuse=tf.AUTO_REUSE):
            # cells
            cell_fw = HighwayWrapper(DropoutWrapper(LSTMCell(size),
                                                    state_keep_prob=keep_prob))
            cell_bw = HighwayWrapper(DropoutWrapper(LSTMCell(size),
                                                    state_keep_prob=keep_prob))

            # forward direction
            with vs.variable_scope("fw") as fw_scope:
                output_fw, output_state_fw = tf.nn.dynamic_rnn(cell=cell_fw,
                                                               inputs=inputs,
                                                               sequence_length=lengths,
                                                               scope=fw_scope,
                                                               dtype=tf.float32)

            # Backward direction
            with vs.variable_scope("bw") as bw_scope:
                inputs_reverse = _reverse(inputs, seq_lengths=lengths, seq_dim=1, batch_dim=0)
                tmp, output_state_bw = tf.nn.dynamic_rnn(cell=cell_bw,
                                                         inputs=inputs_reverse,
                                                         sequence_length=lengths,
                                                         scope=bw_scope,
                                                         dtype=tf.float32)

        # calc either fw or bw - interleaving is done at graph construction (not runtime)
        if layer % 2:
            outputs = _reverse(tmp, seq_lengths=lengths, seq_dim=1, batch_dim=0)
        else:
            outputs = output_fw

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


            # final_outputs = bidirectional_lstms_stacked(inputs, config.num_layers, config.cell_size, self.keep_prob, self.lengths)
            final_outputs = bidirectional_lstms_interleaved(inputs, config.num_layers, config.cell_size, self.keep_prob, self.lengths)

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

