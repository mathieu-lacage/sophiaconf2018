import optparse
import Utils
import json
import collections


def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    parser.add_option('--min-df', type='int', default=None)
    options, args = parser.parse_args()

    documents = Utils.read_json('%s-tokenized.json' % options.dataset)

    n = 0
    df = collections.Counter()
    for tokens in documents:
        n += 1
        df.update(set(tokens))

    with open('%s-df.json' % options.dataset, 'w') as o:
        o.write('%s\n' % json.dumps((-1, '', n)))
        for id, (token, df) in enumerate(df.items()):
            if options.min_df is not None and df < options.min_df:
                continue
            o.write('%s\n' % json.dumps((id, token, df)))


main()    
