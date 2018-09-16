import tensorflow as tf
from tensorflow.contrib.rnn import HighwayWrapper, LSTMCell, DropoutWrapper
from tensorflow.python.ops import array_ops
from socket import gethostname

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
                                                    variational_recurrent=True,
                                                    dtype=tf.float32,
                                                    state_keep_prob=keep_prob))
            cell_bw = HighwayWrapper(DropoutWrapper(LSTMCell(size),
                                                    variational_recurrent=True,
                                                    dtype=tf.float32,
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
        direction = 'backw.' if layer % 2 else 'forw.'
        print('Layer {}: Creating {} LSTM'.format(layer, direction))  # backwards if layer odd
        with tf.variable_scope('{}_lstm_{}'.format(direction, layer)):
            # cell
            cell = HighwayWrapper(DropoutWrapper(LSTMCell(size),
                                                 variational_recurrent=True,
                                                 dtype=tf.float32,
                                                 state_keep_prob=keep_prob))
            # calc either bw or fw - interleaving is done at graph construction (not runtime)
            if direction == 'backw.':
                outputs_reverse = _reverse(outputs, seq_lengths=lengths, seq_dim=1, batch_dim=0)
                tmp, _ = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=outputs_reverse,
                                           sequence_length=lengths,
                                           dtype=tf.float32)
                outputs = _reverse(tmp, seq_lengths=lengths, seq_dim=1, batch_dim=0)
            else:
                outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=outputs,
                                               sequence_length=lengths,
                                               dtype=tf.float32)

    return outputs


class Model():
    def __init__(self, config, embeddings, num_labels, g):

        # embedding
        with tf.device('/cpu:0'):
            self.word_ids = tf.placeholder(tf.int32, [None, None], name='word_ids')
            embedded = tf.nn.embedding_lookup(embeddings, self.word_ids, name='embedded')

        # stacked bilstm
        with tf.device('/gpu:0'):
            self.predicate_ids = tf.placeholder(tf.float32, [None, None], name='predicate_ids')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
            inputs = tf.concat([embedded, tf.expand_dims(self.predicate_ids, -1)], axis=2, name='lstm_inputs')

            if config.architecture == 'stacked':
                final_outputs = bidirectional_lstms_stacked(inputs, config.num_layers, config.cell_size, self.keep_prob, self.lengths)
            elif config.architecture  == 'interleaved':
                final_outputs = bidirectional_lstms_interleaved(inputs, config.num_layers, config.cell_size, self.keep_prob, self.lengths)
            else:
                raise AttributeError('Invalid arg to "architecture"')

            # projection
            shape0 = tf.shape(final_outputs)[0] * tf.shape(final_outputs)[1]  # both batch_size and seq_len are dynamic
            final_outputs_2d = tf.reshape(final_outputs, [shape0, config.cell_size])  # need [shape0, cell_size]
            wy = tf.get_variable('Wy', [config.cell_size, num_labels])
            by = tf.get_variable('by', [num_labels])
            logits = tf.nn.xw_plus_b(final_outputs_2d, wy, by, name='logits')  # need [shape0, num_labels]

            # loss
            self.label_ids = tf.placeholder(tf.int32, [None, None], name='label_ids')  # [batch_size, max_seq_len]
            label_ids_flat = tf.reshape(self.label_ids, [-1])  # need [shape0]


            # TODO test
            mask = tf.greater(label_ids_flat, 0, 'mask')
            nonzero_label_ids_flat = tf.boolean_mask(label_ids_flat, mask, name='nonzero_label_ids_flat')  # removes elements
            nonzero_logits = tf.boolean_mask(logits, mask, name='nonzero_logits')
            nonzero_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nonzero_logits,
                                                                            labels=nonzero_label_ids_flat,
                                                                            name='nonzero_losses')
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=label_ids_flat,
                                                                    name='losses')
            self.nonzero_mean_loss = tf.reduce_mean(nonzero_losses, name='nonzero_mean_loss')
            self.mean_loss = tf.reduce_mean(losses, name='mean_loss')
            # TODO do not mask zero label - make 2d mask in numpy, then flatten and feed into graph

            # update
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.learning_rate, epsilon=config.epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.nonzero_mean_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
            self.update = optimizer.apply_gradients(zip(gradients, variables), name='update')

            # predictions
            predicted_label_ids = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), tf.int32)
            nonzero_predicted_label_ids = tf.cast(tf.argmax(tf.nn.softmax(nonzero_logits), axis=1), tf.int32)
            self.predictions = tf.reshape(predicted_label_ids, tf.shape(self.label_ids))

            # tensorboard
            tf.summary.scalar('accuracy', tf.reduce_mean(tf.cast(tf.equal(predicted_label_ids,
                                                                          label_ids_flat), tf.float32)))
            tf.summary.scalar('nonzero_accuracy', tf.reduce_mean(tf.cast(tf.equal(nonzero_predicted_label_ids,
                                                                                  nonzero_label_ids_flat), tf.float32)))
            tf.summary.scalar('mean_xe', self.nonzero_mean_loss)
            tf.summary.scalar('nonzero_mean_xe', self.nonzero_mean_loss)
            self.merged1 = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('tb/' + gethostname(), g)

            # confusion matrix
            nonzero_cm = tf.confusion_matrix(nonzero_label_ids_flat, nonzero_predicted_label_ids)
            cm = tf.confusion_matrix(label_ids_flat, predicted_label_ids)  # TODO test
            size = tf.shape(cm)[0]
            nonzero_cm_image = tf.summary.image('nonzero_cm', tf.reshape(tf.cast(nonzero_cm, tf.float32),
                                                                              [1, size, size, 1]))  # needs 4d
            cm_image = tf.summary.image('cm', tf.reshape(tf.cast(cm, tf.float32),
                                                              [1, size, size, 1]))
            self.merged2 = tf.summary.merge([nonzero_cm_image, cm_image])




