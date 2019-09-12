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


class Eval:
    loss_interval = 100

    train_data_path = RemoteDirs.data / 'CONLL05/conll05.train.txt'
    dev_data_path = RemoteDirs.data / 'CONLL05/conll05.dev.txt'
    glove_path = RemoteDirs.data / 'glove.6B.100d.txt'