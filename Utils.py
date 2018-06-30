import json
import scipy.io
import scipy.sparse

def tweet_text(tweet):
    if tweet['truncated']:
        return tweet['extended_tweet']['full_text']
    else:
        return tweet['text']


def read_json(filename):
    with open(filename) as f:
        for line in f:
            d = json.loads(line)
            yield d

def write_matrix(filename, m):
    scipy.io.mmwrite(filename, m)

def read_matrix(filename):
    m = scipy.io.mmread(filename)
    return m

class SparseMatrix:
    def __init__(self):
        self._indptr = [0]
        self._indices = []
        self._data = []
    def add_row(self, columns):
        for index, data in columns:
            self._indices.append(index)
            self._data.append(data)
        self._indptr.append(len(self._indices))
    def convert(self):
        matrix = scipy.sparse.csr_matrix((self._data, self._indices, self._indptr))
        return matrix
    def write(self, filename):
        write_matrix(filename, self.convert())
        
