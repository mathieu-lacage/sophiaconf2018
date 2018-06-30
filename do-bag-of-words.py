import optparse
import Utils
import collections
import math


def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    parser.add_option('--mode', default='boolean')
    options, args = parser.parse_args()

    documents = Utils.read_json('%s-tokenized.json' % options.dataset)

    df = dict((token, (id, df)) for id, token, df in Utils.read_json('%s-df.json' % options.dataset))

    m = Utils.SparseMatrix()
    if options.mode == 'boolean':
        for tokens in documents:
            row = [(df[token][0], 1) for token in set(tokens) if token in df]
            m.add_row(row)
    elif options.mode == 'tf':
        for tokens in documents:
            c = collections.Counter(tokens)
            row = [(df[token][0], tf) for token, tf in c.most_common() if token in df]
            m.add_row(row) 
    elif options.mode == 'log-tf':
        for tokens in documents:
            c = collections.Counter(tokens)
            row = [(df[token][0], 1+math.log(tf)) for token, tf in c.most_common() if token in df]
            m.add_row(row) 
    elif options.mode == 'tf-idf':
        for tokens in documents:
            c = collections.Counter(tokens)
            row = [(df[token][0], tf / (1+math.log(df[token][1]))) for token, tf in c.most_common() if token in df]
            m.add_row(row) 
    elif options.mode == 'log-tf-idf':
        for tokens in documents:
            c = collections.Counter(tokens)
            row = [(df[token][0], (1+math.log(tf)) / (1+math.log(df[token][1]))) for token, tf in c.most_common() if token in df]
            m.add_row(row) 
    m.write('%s-vectors.mtx' % options.dataset)

main()    
