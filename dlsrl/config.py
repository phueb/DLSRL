from pathlib import Path
import sys

if 'win' in sys.platform:
    raise SystemExit('Not supported on Windows')
elif 'linux' == sys.platform:
    mnt_point = '/media'
else:
    # assume MacOS
    mnt_point = '/Volumes'


class RemoteDirs:
    research_data = Path(mnt_point) / 'research_data'
    root = research_data / 'DLSRL'
    runs = root / 'runs'
    data = root / 'data'

    train_data = data / 'CONLL05/conll05.train.txt'
    dev_data = data / 'CONLL05/conll05.dev.txt'
    test_data = data / 'CONLL05/conll05.test.wsj.txt'
    glove = data / 'glove.6B.100d.txt'

    srl_eval_script = root / 'perl' / 'srl-eval.pl'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'dlsrl'
    runs = root / '{}_runs'.format(src.name)
    data = root / 'data'

    train_data = data / 'CONLL05/conll05.train.txt'
    dev_data = data / 'CONLL05/conll05.dev.txt'
    test_data = data / 'CONLL05/conll05.test.wsj.txt'
    glove = data / 'glove.6B.100d.txt'

    srl_eval_script = root / 'perl' / 'srl-eval.pl'


class Global:
    debug = False
    local = False


class Data:
    """
    # the letter "O" is used to mark any words without labels as "outside" in conll05
    """

    lowercase = True  # True gives strong performance boost
    bio_tags = True  # the "O" tag is still used to label "outside" words if set to False

    unk_word = '<UNKNOWN>'  # TODO use this for test data - but don't use an UNKNOWN_LABEL

    pad_word = '<PAD>'
    pad_label = 'B-PAD'  # do not use the letter "O" because evaluation requires only removing padding

    verbose = False  # Data class is no longer used to convert between labels and IDs and tokens and IDs


class Eval:
    loss_interval = 100
    summary_interval = 100
    verbose = True  # print output of perl evaluation script
    dev_batch_size = 512  # too big will cause tensorflow internal error


    ignore_span_metric = True  # TODO test



