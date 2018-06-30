import optparse
import Utils

import numpy
import sklearn.cluster
import json


def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    parser.add_option('--mini-batch', default=False, action='store_true')
    parser.add_option('-k', type='int', default=40)
    options, args = parser.parse_args()

    documents = Utils.read_matrix('%s-vectors.mtx' % options.dataset)
    if options.mini_batch:
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters = options.k)
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters = options.k)
    clusters = kmeans.fit_predict(documents)

    documents_by_cluster = dict((c, list(numpy.where(clusters == c)[0])) for c in range(options.k))

    with open('%s-clusters.json' % options.dataset, 'w') as o:
        for c, documents in documents_by_cluster.items():
            o.write('%s\n' % json.dumps((c, documents))) 

    for c in range(options.k):
        print c, numpy.sum(clusters == c)

main()
