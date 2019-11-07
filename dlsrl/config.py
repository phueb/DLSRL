from pathlib import Path


class LocalDirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    glove = data / 'glove.6B.100d.txt'


class Global:
    debug = False
    local = False


class Data:
    lowercase = True  # True gives strong performance boost
    bio_tags = True  # TODO test False


class Eval:
    loss_interval = 100
    verbose = True  # print output of perl evaluation script