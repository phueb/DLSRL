import numpy as np
import random

from dlsrl import config


def make_word2embed(params):
    glove_p = config.RemoteDirs.root / config.Data.glove_path
    assert str(params.embed_size) in glove_p.name
    print('Loading word embeddings at:')
    print(glove_p)

    res = dict()
    with glove_p.open('rb') as f:
        for line in f:
            info = line.strip().split()
            word = info[0]
            embedding = np.array([float(r) for r in info[1:]])
            res[word] = embedding

    embedding_size = next(iter(res.items()))[1].shape[0]
    print('Glove embedding size={}'.format(embedding_size))
    assert embedding_size < params.cell_size - 1  # random weights will be added to make shape equal to cell_size

    # add random vectors to non-word symbols
    res[config.Data.start_word] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    res[config.Data.end_word] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    if config.Data.unk_word not in res:
        res[config.Data.unk_word] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    if config.Data.pad_word not in res:
        res[config.Data.pad_word] = [random.gauss(0, 0.01) for _ in range(embedding_size)]

    return res


def shuffle_stack_pad(data, batch_size, shuffle=True):
    """
    :param shuffle: whether to shuffle data
    :param batch_size: size batch to use
    :param data: [list of word_id seqs, list of pred_id seqs, list of label_id seqs]
    :return: zero-padded matrices for each list of shape [num_seqs, max_seq_len]
    """
    shape0 = len(data[1])
    num_excluded = shape0 % batch_size
    print('Excluding {} sequences due to fixed batch size'.format(num_excluded))
    shape0_adj = shape0 - num_excluded
    shape1 = np.max([len(i) for i in data[1]])
    mats = [np.zeros((shape0_adj, shape1)).astype(np.int32) for _ in range(3)]
    if shuffle:
        row_ids = np.random.choice(shape0, shape0_adj, replace=False)
    else:
        row_ids = np.arange(shape0_adj)
    for sequences, mat in zip(data, mats):
        for n, rand_id in enumerate(row_ids):
            seq = sequences[rand_id]
            mat[n, :len(seq)] = seq
    return mats  # x1, x2, y


def count_zeros_from_end(s):
    res = 0
    it = iter(s[::-1])
    while next(it) == 0:
        res += 1
    else:
        return res


def get_batches(x1, x2, y, batch_size):
    num_batches = len(x1) // batch_size
    print('Generating {} batches with size {}'.format(num_batches, batch_size))
    for x1_b, x2_b, y_b in zip(np.vsplit(x1, num_batches),
                               np.vsplit(x2, num_batches),
                               np.vsplit(y, num_batches)):

        yield x1_b, x2_b, y_b


