import numpy as np
import random

UNKNOWN_TOKEN = '*UNKNOWN*'
UNKNOWN_LABEL = 'O'


class TaggerData(object):
    def __init__(self,
                 config,
                 train_sents,
                 dev_sents,
                 word_dict,
                 label_dict,
                 embeddings):
        self.max_seq_length = config.max_seq_length
        self.max_dev_length = max([len(s[0]) for s in dev_sents]) if len(dev_sents) > 0 else 0
        self.batch_size = config.batch_size
        self.use_se_marker = config.use_se_marker
        self.unk_id = word_dict.str2idx[UNKNOWN_TOKEN]
        self.word_dict = word_dict
        self.label_dict = label_dict
        self.embeddings = embeddings
        self.train_tensors = [self.tensorize(s, self.max_seq_length) for s in train_sents
                              if len(s[0]) <= self.max_seq_length]
        self.dev_tensors = [self.tensorize(s, self.max_dev_length) for s in dev_sents]

    def tensorize(self, sentence, max_length):
        """ Input:
            - sentence: The sentence is a tuple of lists (s1, s2, ..., sk)
                  s1 is always a sequence of word ids.
                  sk is always a sequence of label ids.
                  s2 ... sk-1 are sequences of feature ids,
                    such as predicate or supertag features.
            - max_length: The maximum length of sequences, used for padding.
        """
        x1 = np.array(sentence[0])
        x2 = np.array(sentence[1])  # ph: hardcoded predicate feature
        y = np.array(sentence[-1])
        mask = (y >= 0).astype(float)
        x1.resize([max_length])
        x2.resize([max_length])
        y.resize([max_length])
        mask.resize([max_length])
        return x1, x2, np.absolute(y), mask

    def get_batched_tensors(self, which='train'):  # TODO implement variable sized batches
        if which == 'test':
            raise NotImplemented
        elif which == 'dev':
            num_tensors = len(self.train_tensors)
            tensors = self.dev_tensors
            print('Batching dev tensors...')
        else:
            num_tensors = len(self.train_tensors)
            # shuffling
            shuffled_ids = np.arange(num_tensors)
            random.shuffle(shuffled_ids)
            tensors = [self.train_tensors[t] for t in shuffled_ids]
            print('Batching train tensors...')
        # batching
        batched_tensors = [tensors[i: min(i + self.batch_size, num_tensors)]
                           for i in range(0, num_tensors, self.batch_size)]
        print("Extracted {} samples and {} batches.".format(num_tensors, len(batched_tensors)))
        for b in batched_tensors:
            word_id_batch, f_id_batch, y_batch, mask_batch = list(zip(*b))

            # to get x_batch dim= [batch_size, seq_len, 2], x has to be [16,2]

            print('batched')
            print(np.array(word_id_batch).shape)
            print(np.array(f_id_batch).shape)
            print(np.array(y_batch).shape)
            print(np.array(mask_batch).shape)

            yield np.array(word_id_batch), np.array(f_id_batch), np.array(y_batch), np.array(mask_batch)

  
