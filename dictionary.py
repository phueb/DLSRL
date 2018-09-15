''' Bidirectional dictionary that maps between words and ids.
'''


class Dictionary(object):
    def __init__(self, unknown_token=None):
        self.str2idx = {}
        self.idx2str = []
        self.accept_new = True
        self.unknown_token = None
        self.unknown_id = None
        if unknown_token is not None:
            self.set_unknown_token(unknown_token)

    def set_unknown_token(self, unknown_token):
        self.unknown_token = unknown_token
        self.unknown_id = self.add(unknown_token)

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