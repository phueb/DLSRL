from pathlib import Path


class RemoteDirs:
    root = Path('/media/research_data') / 'DLSRL'
    runs = root / 'runs'
    data = root / 'data'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'dlsrl'
    runs = root / '{}_runs'.format(src.name)
    data = root / 'data'


class Global:
    debug = False


class Data:
    """
    # the letter "O" is used to mark any words without labels as "outside" in conll05
    """

    lowercase = True  # True gives strong performance boost

    start_word = '<S>'
    start_label = 'START_LABEL'  # TODO do not predict these

    end_word = '</S>'
    end_label = 'END_LABEL'

    unk_word = '<UNKNOWN>'
    unk_label = 'UNK_LABEL'

    pad_word = '<PAD>'
    pad_label = 'O'  # using the letter "O" because it is used to signal unlabeled word

    train_data_path = RemoteDirs.data / 'CONLL05/conll05.train.txt'
    dev_data_path = RemoteDirs.data / 'CONLL05/conll05.dev.txt'
    test_data_path = RemoteDirs.data / 'CONLL05/conll05.test.wsj.txt'
    glove_path = RemoteDirs.data / 'glove.6B.100d.txt'

    verbose = True


class Eval:
    loss_interval = 100
    summary_interval = 100
    verbose = False

    use_bio_tags = False  # TODO testing

