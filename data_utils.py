import random
import numpy as np
import pickle

from dictionary import Dictionary

WORD_EMBEDDINGS = { "50":  'glove.6B.50d.txt',
                    "100": 'glove.6B.100d.txt',
                    "200": 'glove.6B.200d.txt'}

START_MARKER  = '<S>'
END_MARKER    = '</S>'
UNKNOWN_TOKEN = '*UNKNOWN*'
PADDING_TOKEN = '*PAD*'


def get_propositions_from_file(filepath, use_se_marker=False):
    """
    Read tokenized propositions from file.
      File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
      Return:
        A list with elements of structure [[words], predicate, [labels]]
    """
    propositions = []
    with open(filepath) as f:
        for line in f.readlines():
            inputs = line.strip().split('|||')
            lefthand_input = inputs[0].strip().split()
            # If gold tags are not provided, create a sequence of dummy tags.
            righthand_input = inputs[1].strip().split() if len(inputs) > 1 \
                else ['O' for _ in lefthand_input[1:]]
            predicate = int(lefthand_input[0])
            if use_se_marker:
                words = [START_MARKER] + lefthand_input[1:] + [END_MARKER]
                labels = [None] + righthand_input + [None]
            else:
                words = lefthand_input[1:]
                labels = righthand_input
            propositions.append((words, predicate, labels))
    return propositions


def make_word2embed(filepath):
    # load
    print('Loading embeddings...')
    with open(filepath + '.pkl', 'rb') as f:
        word_to_embed_dict = pickle.load(f)
    # add vectors
    embedding_size = next(iter(word_to_embed_dict.items()))[1].shape[0]
    word_to_embed_dict[START_MARKER] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    word_to_embed_dict[END_MARKER] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    if UNKNOWN_TOKEN not in word_to_embed_dict:
        word_to_embed_dict[UNKNOWN_TOKEN] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    if PADDING_TOKEN not in word_to_embed_dict:
        word_to_embed_dict[PADDING_TOKEN] = [random.gauss(0, 0.01) for _ in range(embedding_size)]
    return word_to_embed_dict


def words_to_ids(str_seq, dictionary, lowercase=False, word2embed=None):
    ids = []
    for s in str_seq:
        if s is None:
            ids.append(-1)
            continue
        if lowercase:
            s = s.lower()
        if word2embed is not None and s not in word2embed:  # if word is not in embedding, make UNKNOWN
            s = UNKNOWN_TOKEN
        ids.append(dictionary.add(s))
    return ids


def get_predicate_ids(propositions, config):
    use_se_marker = config.use_se_marker
    offset = int(use_se_marker)
    predicate_ids = [[int(i == p[1] + offset) for i in range(len(p[0]))] for p in propositions]
    return predicate_ids


def get_data(config, train_data_path, dev_data_path):
    # read sentences from file
    use_se_marker = config.use_se_marker
    raw_train_props = get_propositions_from_file(train_data_path, use_se_marker)
    raw_dev_props = get_propositions_from_file(dev_data_path, use_se_marker)
    word2embed = make_word2embed(WORD_EMBEDDINGS[config.embed_size])

    # TODO data is correct until here

    # prepare word dictionary
    word_dict = Dictionary()  # do not add words from test data to word_dict
    word_dict.set_padding_token(PADDING_TOKEN)  # must be first to get zero id
    word_dict.set_unknown_token(UNKNOWN_TOKEN)
    if use_se_marker:
        word_dict.add(START_MARKER)
        word_dict.add(END_MARKER)

    # prepare label dictionary
    label_dict = Dictionary()
    label_dict.set_unknown_token('O')  # set this first to guarantee id of zero

    # train_data
    train_word_ids = [words_to_ids(p[0], word_dict, True, word2embed) for p in raw_train_props]
    train_predicate_ids = get_predicate_ids(raw_train_props, config)
    train_label_ids = [words_to_ids(p[2], label_dict) for p in raw_train_props]
    train_data = [train_word_ids, train_predicate_ids, train_label_ids]

    # TODO data is correct until here

    label_dict.accept_new = False
    num_labels = label_dict.size()

    # dev_data
    dev_word_ids = [words_to_ids(p[0], word_dict, True, word2embed) for p in raw_dev_props]
    dev_predicate_ids = get_predicate_ids(raw_dev_props, config)
    dev_label_ids = [words_to_ids(p[2], label_dict) for p in raw_dev_props]
    dev_data = [dev_word_ids, dev_predicate_ids, dev_label_ids]

    # print
    print('/////////////////////////////')
    print('Found {:,} training propositions ...'.format(len(raw_train_props)))
    print('Found {:,} dev propositions ...'.format(len(raw_dev_props)))
    print("Extracted {:,} train+dev words and {:,} tags".format(word_dict.size(), num_labels))
    for which, data in zip(['train', 'dev'], [train_data, dev_data]):
        print("Max {} sentence length: {}".format(which, np.max([len(i) for i in data[0]])))
        print("Mean {} sentence length: {}".format(which, np.mean([len(i) for i in data[0]])))
        print("Median {} sentence length: {}".format(which, np.median([len(i) for i in data[0]])))
    print('/////////////////////////////')

    # resize embeddings to enable residual connections
    emb_shape1 = config.cell_size - 1  # -1 because of predicate_id
    print('Reshaping embedding dim1 to {}'.format(emb_shape1))
    embeddings = np.random.normal(0, scale=0.001, size=[word_dict.size(), emb_shape1]).astype(np.float32)
    for n, w in enumerate(word_dict.idx2str):
        embedding = word2embed[w]
        embeddings[n, :len(embedding)] = embedding

    return (train_data,
            dev_data,
            word_dict,
            label_dict,
            embeddings)
