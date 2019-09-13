import tensorflow as tf


def orthonorm(shape, dtype=tf.float32,  # TODO only works for square (recurrent) weights
              partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.linalg.svd(tf.random.normal(shape, dtype=dtype), full_matrices=True)
    return u


class Model:
    def __init__(self, params, embeddings, num_labels):
        self.params = params
        self.embeddings = embeddings
        self.num_labels = num_labels

        self.lstm_cells = [tf.keras.layers.LSTM(self.params.cell_size,
                                                activation='tanh',
                                                recurrent_activation='sigmoid',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                recurrent_initializer='orthogonal',  # TODO orthonormal?
                                                dropout=1 - self.params.keep_prob,
                                                recurrent_dropout=0.0,
                                                go_backwards=True if layer % 2 else False)
                           for layer in params.num_layers]
        self.output = tf.keras.layers.Dense(self.num_labels, input_shape=(None, self.params.cell_size))
        self.optimizer = tf.optimizers.Adadelta(learning_rate=params.learning_rate, epsilon=params.epsilon)

    @tf.function
    def embed(self, word_ids):
        """

        :param word_ids: int32, [batch_size, max_seq_len]
        :return: float 32, [batch_size, max_seq_len, cell_size -1]
        """
        res = tf.nn.embedding_lookup(params=self.embeddings, ids=word_ids)
        return res

    @tf.function
    def encode_with_lstm(self, embedded, predicate_ids, mask):  # TODO add highway connections + dropout
        """

        :param embedded: float32, [batch_size, cell_size - 1]
        :param predicate_ids: float32, [batch_size, 1]
        :param mask: int32, [batch_size, max_seq_len] binary tensor indicating if a step should be masked
        :return:
        """

        concatenated = tf.concat([embedded, tf.expand_dims(predicate_ids, -1)], axis=2)

        outputs = concatenated

        for layer, lstm_cell in enumerate(self.lstm_cells):
            print('Using LSTM cell in layer {}'.format(layer))

            # TODO go_backwards

            lstm_cell = self.lstm_cells[layer]
            print('made lstm cell. calling it')
            print(outputs.shape)
            print(mask.shape)

            outputs = lstm_cell(inputs=outputs, mask=mask)

        return outputs

    def calc_logits(self, encoded):
        """
        
        :param encoded: float32, [batch_size, cell_size]
        :return: float32, [batch_size, num_labels]
        """
        shape0 = tf.shape(input=encoded)[0] * tf.shape(input=encoded)[1]
        encoded_2d = tf.reshape(encoded, [shape0, self.params.cell_size])
        res = self.output(encoded_2d)  # need [shape0, num_labels]
        return res

    @tf.function
    def calc_loss(self, logits, label_ids):
        """

        :param logits:
        :param label_ids: int32, [batch_size, max_seq_len]
        :return:
        """

        label_ids_flat = tf.reshape(label_ids, [-1])  # need [shape0]
        mask = tf.greater(label_ids_flat, 0)
        nonzero_label_ids_flat = tf.boolean_mask(tensor=label_ids_flat, mask=mask)  # removes elements
        nonzero_logits = tf.boolean_mask(tensor=logits, mask=mask)
        nonzero_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nonzero_logits,
                                                                        labels=nonzero_label_ids_flat)
        nonzero_mean_loss = tf.reduce_mean(input_tensor=nonzero_losses)
        return nonzero_mean_loss

    @tf.function
    def update(self):
        gradients, variables = zip(*self.optimizer.compute_gradients(self.nonzero_mean_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.params.max_grad_norm)
        self.optimizer.apply_gradients(zip(gradients, variables), name='update')

    @tf.function
    def predict(self, nonzero_logits):
        nonzero_predicted_label_ids = tf.cast(tf.argmax(input=tf.nn.softmax(nonzero_logits), axis=1), tf.int32)
        return nonzero_predicted_label_ids

    def write_summaries(self):
        tf.compat.v1.summary.scalar('nonzero_accuracy',
                                    tf.reduce_mean(input_tensor=tf.cast(tf.equal(self.nonzero_predicted_label_ids,
                                                                                 self.nonzero_label_ids_flat),
                                                                        tf.float32)))
        tf.compat.v1.summary.scalar('nonzero_mean_xe', self.nonzero_mean_loss)
        self.scalar_summaries = tf.compat.v1.summary.merge_all()

        # confusion matrix
        nonzero_cm = tf.math.confusion_matrix(labels=self.nonzero_label_ids_flat,
                                              predictions=self.nonzero_predicted_label_ids)
        size = tf.shape(input=nonzero_cm)[0]
        self.cm_summary = tf.compat.v1.summary.image('nonzero_cm', tf.reshape(tf.cast(nonzero_cm, tf.float32),
                                                                              [1, size, size, 1]))  # needs 4d

        # TODO do the actual writing
