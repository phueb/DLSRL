from collections import OrderedDict
import numpy as np

from dlsrl import config


class Data:

    def __init__(self, params, w2embed):
        self.params = params
        self.w2embed = w2embed

        # ----------------------------------------------------------- words & labels

        self.word_set = set()  # holds words from both train and dev
        self.label_set = set()  # holds labels from both train and dev

        self.train_propositions = self.get_propositions_from_file(config.Data.train_data_path)
        self.dev_propositions = self.get_propositions_from_file(config.Data.dev_data_path)

        self.sorted_words = sorted(self.word_set)
        self.sorted_labels = sorted(self.label_set)

        self.sorted_words = [config.Data.pad_word, config.Data.unk_word] + self.sorted_words  # pad must have id=0
        self.sorted_labels = [config.Data.pad_label, config.Data.unk_label] + self.sorted_labels

        if params.use_se_marker:
            self.sorted_words += [config.Data.start_word, config.Data.end_word]
            self.sorted_labels+= [config.Data.start_label, config.Data.end_label]

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
        print("Extracted {:,} train+dev words and {:,} tags".format(self.num_words, self.num_labels))

        for name, propositions in zip(['train', 'dev'],
                               [self.train_propositions, self.dev_propositions]):
            lengths = [len(i) for i in propositions[0]]
            print("Max {} sentence length: {}".format(name, np.max(lengths)))
            print("Mean {} sentence length: {}".format(name, np.mean(lengths)))
            print("Median {} sentence length: {}".format(name, np.median(lengths)))
        print('/////////////////////////////')

        # -------------------------------------------------------- prepare data structures for training

        self.train_predicate_ids = self.get_predicate_ids(self.train_propositions)
        self.dev_predicate_ids = self.get_predicate_ids(self.dev_propositions)

        # TODO make infinite generator (yields batches of ids) - or make tf.data.Dataset?


    @property
    def num_labels(self):
        return len(self.label_set)

    @property
    def num_words(self):
        return len(self.word_set)

    @property
    def num_train_propositions(self):
        return len(self.train_propositions)

    @property
    def num_dev_propositions(self):
        return len(self.train_propositions)

    def make_embeddings(self):  # TODO also filter out words that are not in vocab + how to order?

        # TODO deal with words not in glove embeddings


        # resize embeddings to enable residual connections
        new_dim1 = self.params.cell_size - 1  # -1 because of predicate_id
        print('Reshaping embedding dim1 to {}'.format(new_dim1))
        random_normals = np.random.normal(0, scale=0.001, size=[self.num_words, new_dim1]).astype(np.float32)
        for n, w in enumerate(word_dict.idx2str):
            embedding = word2embed[w]
            random_normals[n, :len(embedding)] = embedding

        return res  # TODO test

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

                self.word_set.update(words)
                self.label_set.update(labels)

                # TODO collect ids here too?

                propositions.append((words, predicate, labels))

        return propositions

    def get_predicate_ids(self, propositions):
        offset = int(self.params.use_se_marker)
        predicate_ids = [[int(i == p[1] + offset) for i in range(len(p[0]))] for p in propositions]
        return predicate_ids

    @property
    def train_data(self):
        return [self.train_word_ids,
                self.train_predicate_ids,
                self.train_label_ids]

    @property
    def dev_data(self):
        return [self.dev_word_ids,
                self.dev_predicate_ids,
                self.dev_label_ids]


