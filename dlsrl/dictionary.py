""" Bidirectional dictionary that maps between words and ids.
"""


class Dictionary(object):
    def __init__(self):
        self.str2idx = {}
        self.idx2str = []
        self.accept_new = True
        self.unknown_str = None
        self.unknown_id = None
        self.padding_str = None
        self.padding_id = None

    def set_unknown_str(self, s):
        self.unknown_str = s
        self.unknown_id = self.add(s)

    def set_padding_str(self, s):
        self.padding_str = s
        self.padding_id = self.add(s)

    def add(self, new_str):
        if new_str not in self.str2idx:
            if self.accept_new:
                self.str2idx[new_str] = len(self.idx2str)
                self.idx2str.append(new_str)
            else:
                if self.unknown_id is None:
                    raise LookupError(
                        'Trying to add new token to a freezed dictionary with no pre-defined unknown token: ' + new_str)
                return self.unknown_id

        return self.str2idx[new_str]

    def size(self):
        return len(self.idx2str)