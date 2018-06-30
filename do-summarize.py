import optparse
import Utils
import collections
import math

def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    options, args = parser.parse_args()

    documents = list(Utils.read_json('%s-tokenized.json' % options.dataset))
    df = dict((token, df) for id, token, df in Utils.read_json('%s-df.json' % options.dataset))
    
    clusters = list(Utils.read_json('%s-clusters.json' % options.dataset))

    for cluster, documents_ids in clusters:
        cluster_token_tf = collections.Counter()
        for document_id in documents_ids:
            document = documents[document_id]
            cluster_token_tf.update(set(document))
        kl_divergence = [(math.log(float(tf) / len(documents_ids)) - math.log(float(df[token]) / df['']), float(tf) / len(documents_ids), token) for token, tf in cluster_token_tf.most_common() if token in df]
        most_relevant = sorted(kl_divergence, key = lambda i: i[0:2], reverse = True)[:5]
        print('%d\t%d\t%s' % (cluster, len(documents_ids), ' '.join([token for kl, tf, token in most_relevant])))

main()
