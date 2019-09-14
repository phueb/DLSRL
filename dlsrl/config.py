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
    lowercase = True  # TODO use

    start_word = '<S>'
    start_label = 'START_LABEL'  # TODO do not predict these

    end_word = '</S>'
    end_label = 'END_LABEL'

    unk_word = '*UNKNOWN*'
    unk_label = 'O'  # must match CONLL05 file

    pad_word = '*PAD*'
    pad_label = 'PAD_LABEL'

    train_data_path = RemoteDirs.data / 'CONLL05/conll05.train.txt'
    dev_data_path = RemoteDirs.data / 'CONLL05/conll05.dev.txt'
    glove_path = RemoteDirs.data / 'glove.6B.100d.txt'


class Eval:
    summary_interval = 100


