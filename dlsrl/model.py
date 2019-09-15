import tensorflow as tf


def orthonorm(shape, dtype=tf.float32,  # TODO only works for square (recurrent) weights
              partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.linalg.svd(tf.random.normal(shape, dtype=dtype), full_matrices=True)
    return u


class Model(tf.keras.Model):

    def __init__(self, params, embeddings, num_labels, name='deep_lstm', **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

        print('Initializing keras model with in size = {} and out size = {}'.format(
            len(embeddings), num_labels))

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
                                                return_sequences=True,  # TODO test
                                                go_backwards=True if layer % 2 else False)
                           for layer in range(params.num_layers)]
        self.to_logits = tf.keras.layers.Dense(self.num_labels,
                                               input_shape=(None, self.params.cell_size),
                                               activation=None)

    def call(self, word_ids, predicate_ids, mask):
        embedded = self.embed(word_ids)
        encoded = self.encode_with_lstm(embedded, predicate_ids, mask)
        logits = self.calc_logits(encoded)
        return logits

    # TODO ---------------------------------------------------- convert below to tf.keras Layers

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
        :return: float32, [batch_size, max_seq_len, cell_size]
        """

        concatenated = tf.concat([embedded,
                                  tf.expand_dims(tf.cast(predicate_ids, tf.float32), -1)], axis=2)

        res = concatenated

        for layer, lstm_cell in enumerate(self.lstm_cells):
            lstm_cell = self.lstm_cells[layer]
            res = lstm_cell(inputs=res, mask=tf.cast(mask, tf.bool))  # [batch_size, max_seq_len, cell_size]

        return res

    def calc_logits(self, encoded):
        """
        
        :param encoded: float32, [batch_size, max_seq_len, cell_size]
        :return: float32, [num_words, num_labels]
        """
        num_words = tf.shape(input=encoded)[0] * tf.shape(input=encoded)[1]
        encoded_2d = tf.reshape(encoded, [num_words, self.params.cell_size])
        res = self.to_logits(encoded_2d)  # need [num_words, num_labels]
        return res

    # ------------------------------------------------------- unrelated to training

    @tf.function
    def predict_label_ids(self, word_ids, predicate_ids, mask):
        """

        :param word_ids:
        :param predicate_ids:
        :param mask:
        :return: label IDs (one for each word ID), [num_words, num_labels]
        """
        embedded = self.embed(word_ids)
        encoded = self.encode_with_lstm(embedded, predicate_ids, mask)
        logits = self.calc_logits(encoded)
        res = tf.cast(tf.argmax(input=tf.nn.softmax(logits), axis=1), tf.int32)
        return res

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

        # TODO https://www.tensorflow.org/tensorboard/r2/get_started