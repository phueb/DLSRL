from pathlib import Path


class RemoteDirs:
    root = Path('/media/research_data') / 'DLSRL'
    runs = root / 'runs'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'dlsrl'
    runs = root / '{}_runs'.format(src.name)


class Global:
    debug = False


class Eval:
    loss_interval = 100

    train_data_path = 'CONLL05/conll05.train.txt'
    dev_data_path = 'CONLL05/conll05.dev.txt'
