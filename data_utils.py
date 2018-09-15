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


def get_sentences_from_file(filepath, use_se_marker=False):
    """ Read tokenized SRL sentences from file.
      File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
      Return:
        A list of sentences, with structure: [[words], predicate, [labels]]
    """
    sentences = []
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
            sentences.append((words, predicate, labels))
    return sentences


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
    return word_to_embed_dict


def string_sequence_to_ids(str_seq, dictionary, lowercase=False, word2embed=None):
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


def get_predicate_ids(sentences, config):
    use_se_marker = config.use_se_marker
    offset = int(use_se_marker)
    predicate_ids = [[int(i == sent[1] + offset) for i in range(len(sent[0]))] for sent in sentences]
    return predicate_ids


def get_data(config, train_data_path, dev_data_path):
    # read sentences from file
    use_se_marker = config.use_se_marker
    raw_train_sents = get_sentences_from_file(train_data_path, use_se_marker)
    raw_dev_sents = get_sentences_from_file(dev_data_path, use_se_marker)
    word2embed = make_word2embed(WORD_EMBEDDINGS[config.embed_size])

    # prepare word dictionary.
    word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)  # do not add words from test data to word_dict
    if use_se_marker:
        word_dict.add(START_MARKER)
        word_dict.add(END_MARKER)

    # prepare label dictionary.
    label_dict = Dictionary()
    label_dict.set_unknown_token('O')  # set this first to guarantee id of zero

    # train_data
    train_word_ids = [string_sequence_to_ids(sent[0], word_dict, True, word2embed) for sent in raw_train_sents]
    train_predicate_ids = get_predicate_ids(raw_train_sents, config)
    train_label_ids = [string_sequence_to_ids(sent[2], label_dict) for sent in raw_train_sents]
    train_data = [np.array(train_word_ids), np.array(train_predicate_ids), np.array(train_label_ids)]

    if label_dict.accept_new:

        label_dict.accept_new = False

    # dev_data
    dev_word_ids = [string_sequence_to_ids(sent[0], word_dict, True, word2embed) for sent in raw_dev_sents]
    dev_predicate_ids = get_predicate_ids(raw_dev_sents, config)
    dev_label_ids = [string_sequence_to_ids(sent[2], label_dict) for sent in raw_dev_sents]
    dev_data = [np.array(dev_word_ids), np.array(dev_predicate_ids), np.array(dev_label_ids)]  # TODO test

    num_labels = label_dict.size()  # TODO make sure this is correct - is softmax shifted over by one?

    # TODO test zero label
    print('Label dict:')
    for k, v in label_dict.str2idx.items():
        print('"{}" has id {}'.format(k, v))

    print('/////////////////////////////')
    print('Found {:,} training sentences ...'.format(len(raw_train_sents)))
    print('Found {:,} dev sentences ...'.format(len(raw_dev_sents)))
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
            num_labels,
            embeddings)
