''' Bidirectional dictionary that maps between words and ids.
'''


class Dictionary(object):
    def __init__(self):
        self.str2idx = {}
        self.idx2str = []
        self.accept_new = True
        self.unknown_token = None
        self.unknown_id = None
        self.padding_token = None

    def set_unknown_token(self, unknown_token):
        self.unknown_token = unknown_token
        self.unknown_id = self.add(unknown_token)

    def set_padding_token(self, padding_token):
        self.padding_token = padding_token
        self.padding_id = self.add(padding_token)

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