import numpy as np
import random

UNKNOWN_TOKEN = '*UNKNOWN*'
UNKNOWN_LABEL = 'O'


class Batcher():
    def __init__(self,
                 config,
                 train_data,
                 dev_data):
        self.dev_batch_size = config.dev_batch_size
        self.train_batch_size = config.train_batch_size
        self.max_seq_length = config.max_seq_length
        self.max_dev_length = max([len(s[0]) for s in dev_data]) if len(dev_data) > 0 else 0
        self.use_se_marker = config.use_se_marker
        self.train_tensors = [self.tensorize(s, self.max_seq_length) for s in train_data
                              if len(s[0]) <= self.max_seq_length]
        self.dev_tensors = [self.tensorize(s, self.max_dev_length) for s in dev_data]

    def tensorize(self, data, max_length):
        """ Input:
            - data is a tuple of lists (x1, x2, y)
                  x1 is always a sequence of word ids.
                  x2 is always a sequence of predicate ids.
                  y is always sequence of label ids,
            - max_length: The maximum length of sequences, used for padding.
        """
        x1 = np.array(data[0])
        x2 = np.array(data[1])
        y = np.array(data[2])
        x1.resize([max_length])
        x2.resize([max_length])
        y.resize([max_length])
        return x1, x2, np.absolute(y)

    def get_batched_tensors(self, which='train'):  # TODO implement variable sized batches
        if which == 'test':
            raise NotImplemented
        elif which == 'dev':
            num_tensors = len(self.dev_tensors)
            batch_size = self.dev_batch_size
            tensors = self.dev_tensors
            print('Batching dev tensors with batch size {}...'.format(batch_size))
        else:
            num_tensors = len(self.train_tensors)
            batch_size = self.train_batch_size
            # shuffling
            shuffled_ids = np.arange(num_tensors)
            random.shuffle(shuffled_ids)
            tensors = [self.train_tensors[t] for t in shuffled_ids]
            print('Batching train tensors with batch size {}...'.format(batch_size))
        # batching
        batched_tensors = [tensors[i: min(i + batch_size, num_tensors)]
                           for i in range(0, num_tensors, batch_size)][:-1]
        print("Extracted {} samples and {} batches.".format(num_tensors, len(batched_tensors)))
        for n, b in enumerate(batched_tensors):
            word_id_batch, p_id_batch, l_id_batch = list(zip(*b))
            yield np.array(word_id_batch), np.array(p_id_batch), np.array(l_id_batch)

  
