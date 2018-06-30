import optparse
import Utils
import gensim
import numpy

def doc2vec(m, document):
    vectors = [m[token] for token in document if token in m]
    if len(vectors) == 0:
        return None
    doc = numpy.sum(vectors, axis=0)
    return doc


def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    options, args = parser.parse_args()

    documents = list(Utils.read_json('%s-tokenized.json' % options.dataset))

    m = gensim.models.word2vec.Word2Vec.load('%s-word-vector-model' % options.dataset)

    vectors = [doc2vec(m, document) for document in documents]
    vectors = [v if v is not None else numpy.zeros_like(vectors[0]) for v in vectors]
    vectors = numpy.array(vectors)

    Utils.write_matrix('%s-vectors.mtx' % options.dataset, vectors)

main()    
