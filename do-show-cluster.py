import optparse
import Utils
import collections
import math

def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    options, args = parser.parse_args()

    cluster_id = int(args[0])

    documents = list(Utils.read_json('%s-tokenized.json' % options.dataset))
    
    clusters = list(Utils.read_json('%s-clusters.json' % options.dataset))

    for cluster, documents_ids in clusters:
        if cluster != cluster_id:
            continue
        for document_id in documents_ids:
            print documents[document_id]

main()
