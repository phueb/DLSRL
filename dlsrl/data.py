from collections import OrderedDict
import numpy as np

from dlsrl import config


class Data:

    def __init__(self, params):
        self.params = params

        # ----------------------------------------------------------- words & labels

        self._word_set = set()  # holds words from both train and dev
        self._label_set = set()  # holds labels from both train and dev

        self.train_propositions = self.get_propositions_from_file(config.Data.train_data_path)
        self.dev_propositions = self.get_propositions_from_file(config.Data.dev_data_path)

        self.sorted_words = sorted(self._word_set)
        self.sorted_labels = sorted(self._label_set)

        self.sorted_words = [config.Data.pad_word, config.Data.unk_word] + self.sorted_words  # pad must have id=0
        self.sorted_labels = [config.Data.pad_label, config.Data.unk_label] + self.sorted_labels

        if params.use_se_marker:
            self.sorted_words += [config.Data.start_word, config.Data.end_word]
            self.sorted_labels += [config.Data.start_label, config.Data.end_label]

        self.w2id = OrderedDict()  # word -> ID
        for n, w in enumerate(self.sorted_words):
            self.w2id[w] = n

        self.l2id = OrderedDict()  # label -> ID
        for n, l in enumerate(self.sorted_labels):
            self.l2id[l] = n

        # -------------------------------------------------------- console

        print('/////////////////////////////')
        print('Found {:,} training propositions ...'.format(self.num_train_propositions))
        print('Found {:,} dev propositions ...'.format(self.num_dev_propositions))
        print("Extracted {:,} train+dev words and {:,} labels".format(self.num_words, self.num_labels))

        for name, propositions in zip(['train', 'dev'],
                                      [self.train_propositions, self.dev_propositions]):
            lengths = [len(p[0]) for p in propositions]
            print("Max {} sentence length: {}".format(name, np.max(lengths)))
            print("Mean {} sentence length: {}".format(name, np.mean(lengths)))
            print("Median {} sentence length: {}".format(name, np.median(lengths)))
        print('/////////////////////////////')

        # -------------------------------------------------------- embeddings

        self.embeddings = self.make_embeddings()

        # -------------------------------------------------------- prepare data structures for training

        self.train_data = self.to_ids(self.train_propositions)
        self.dev_data = self.to_ids(self.dev_propositions)

        # TODO make infinite generator (yields batches of ids) - or make tf.data.Dataset?




    @property
    def num_labels(self):
        return len(self.sorted_labels)

    @property
    def num_words(self):
        return len(self.sorted_words)

    @property
    def num_train_propositions(self):
        return len(self.train_propositions)

    @property
    def num_dev_propositions(self):
        return len(self.dev_propositions)

    def get_propositions_from_file(self, file_path):
        """
        Read tokenized propositions from file.
          File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
          Return:
            A list with elements of structure [[words], predicate, [labels]]
        """
        propositions = []
        with file_path.open('r') as f:

            for line in f.readlines():
                inputs = line.strip().split('|||')
                left_input = inputs[0].strip().split()

                # If gold tags are not provided, create a sequence of dummy tags.
                right_input = inputs[1].strip().split() if len(inputs) > 1 \
                    else [config.Data.unk_label for _ in left_input[1:]]

                # predicate
                predicate = int(left_input[0])

                # words + labels
                if self.params.use_se_marker:
                    words = [config.Data.start_word] + left_input[1:] + [config.Data.end_word]
                    labels = [config.Data.start_label] + right_input + [config.Data.end_label]
                else:
                    words = left_input[1:]
                    labels = right_input

                self._word_set.update(words)
                self._label_set.update(labels)

                # TODO where to lower-case?

                propositions.append((words, predicate, labels))

        return propositions

    # ---------------------------------------------------------- embeddings

    def make_embeddings(self):

        assert len(self._word_set) > 0

        glove_p = config.RemoteDirs.root / config.Data.glove_path
        assert str(self.params.embed_size) in glove_p.name
        print('Loading word embeddings at:')
        print(glove_p)

        w2embed = dict()
        with glove_p.open('rb') as f:
            for line in f:
                info = line.strip().split()
                word = info[0]
                embedding = np.array([float(r) for r in info[1:]])
                w2embed[word] = embedding

        embedding_size = next(iter(w2embed.items()))[1].shape[0]
        print('Glove embedding size={}'.format(embedding_size))

        assert len(self.w2id) == self.num_words

        # get embeddings for words in vocabulary
        res = np.zeros((self.num_words, embedding_size), dtype=np.float32)
        for w, row_id in self.w2id.items():
            try:
                res[row_id] = w2embed[w]
            except KeyError:
                res[row_id] = np.random.standard_normal(embedding_size)

        return res

    # --------------------------------------------------------- data structures for training a model

    def make_predicate_ids(self, proposition):
        """

        :param proposition: a tuple with structure (words, predicate, labels)
        :return: one-hot list, [sentence length]
        """
        offset = int(self.params.use_se_marker)
        num_w_in_proposition = len(proposition[0])
        res = [int(i == proposition[1] + offset) for i in range(num_w_in_proposition)]
        return res

    def to_ids(self, propositions):
        """

        :param propositions: a tuple with structure (words, predicate, labels)
        :return: 3 lists, each of the same length, containing lists of integers
        """

        word_ids = []
        predicate_ids = []
        label_ids = []
        for proposition in propositions:
            word_ids.append([self.w2id[w] for w in proposition[0]])
            predicate_ids.append(self.make_predicate_ids(proposition))
            label_ids.append([self.l2id[l] for l in proposition[2]])

        return word_ids, predicate_ids, label_ids