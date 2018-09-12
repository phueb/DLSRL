import random
import re

from dictionary import Dictionary

WORD_EMBEDDINGS = { "glove50":  'glove.6B.50d.txt',
                    "glove100": 'glove.6B.100d.txt',
                    "glove200": 'glove.6B.200d.txt'}

START_MARKER  = '<S>'
END_MARKER    = '</S>'
UNKNOWN_TOKEN = '*UNKNOWN*'
UNKNOWN_LABEL = 'O'


def get_srl_sentences(filepath, use_se_marker=False):
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
    word_to_embed_dict = dict()
    with open(filepath, 'r') as f:
        for line in f:
            info = line.strip().split()
            word = info[0]
            embedding = [float(r) for r in info[1:]]
            word_to_embed_dict[word] = embedding
        f.close()
    embedding_size = int(re.findall('\d+', filepath)[-1])
    print('Embedding size={}'.format(embedding_size))
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
        if (word2embed is not None) and not (s in word2embed) :
            s = UNKNOWN_TOKEN
        ids.append(dictionary.add(s))
    return ids


def get_srl_features(sentences, config):
    feature_names = config.features
    use_se_marker = config.use_se_marker

    features = []
    for fname in feature_names:
        if fname == "predicate":
            offset = int(use_se_marker)
            f = [[int(i == sent[1] + offset) for i in range(len(sent[0]))] for sent in sentences]
            features += f

    return features  # ph: features has another dimension for future features


def get_srl_data(config, train_data_path, dev_data_path, vocab_path=None, label_path=None):
    use_se_marker = config.use_se_marker
    raw_train_sents = get_srl_sentences(train_data_path, use_se_marker)
    raw_dev_sents = get_srl_sentences(dev_data_path, use_se_marker)
    word2embed = make_word2embed(WORD_EMBEDDINGS[config.word_embedding])

    # Prepare word dictionary.
    word_dict = Dictionary(unknown_token=UNKNOWN_TOKEN)
    if use_se_marker:
        word_dict.add_all([START_MARKER, END_MARKER])
    if vocab_path is not None:
        with open(vocab_path, 'r') as f_vocab:
            for line in f_vocab:
                word_dict.add(line.strip())
            f_vocab.close()
        word_dict.accept_new = False

    # prepare label dictionary.
    label_dict = Dictionary()
    if label_path is not None:
        with open(label_path, 'r') as f_labels:
            for line in f_labels:
                label_dict.add(line.strip())
            f_labels.close()
        label_dict.set_unknown_token(UNKNOWN_LABEL)
        label_dict.accept_new = False

    # Get tokens and labels
    train_tokens = [string_sequence_to_ids(sent[0], word_dict, True, word2embed) for sent in raw_train_sents]
    train_labels = [string_sequence_to_ids(sent[2], label_dict) for sent in raw_train_sents]

    if label_dict.accept_new:
        label_dict.set_unknown_token(UNKNOWN_LABEL)
        label_dict.accept_new = False

    dev_tokens = [string_sequence_to_ids(sent[0], word_dict, True, word2embed) for sent in raw_dev_sents]
    dev_labels = [string_sequence_to_ids(sent[2], label_dict) for sent in raw_dev_sents]

    # Get features
    train_features = get_srl_features(raw_train_sents, config)
    dev_features = get_srl_features(raw_dev_sents, config)

    train_sents = [(t, f, l) for t, f, l in zip(train_tokens, train_features, train_labels)]
    dev_sents = [(t, f, l) for t, f, l in zip(dev_tokens, dev_features, dev_labels)]

    print("Extracted {} words and {} tags".format(word_dict.size(), label_dict.size()))
    print("Max training sentence length: {}".format(max([len(s[0]) for s in train_sents])))
    print("Max development sentence length: {}".format(max([len(s[0]) for s in dev_sents])))

    word_embedding = [word2embed[w] for w in word_dict.idx2str]
    return (train_sents,
            dev_sents,
            word_dict,
            label_dict,
            word_embedding)
            # [word_embedding, None, None]) # TODO this is old code, why None?
