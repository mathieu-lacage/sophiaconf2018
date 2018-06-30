import json
import nltk.tokenize
import Utils


def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--dataset', default='sample')
    parser.add_option('--lang', default='fr')
    options, args = parser.parse_args()
    tokenizer = nltk.tokenize.TweetTokenizer(strip_handles = True, reduce_len = True)

    data = Utils.read_json('%s-tweets.json' % options.dataset)

    with open('%s-tokenized.json' % options.dataset, 'w') as o:
        for tweet in data:
            if not 'lang' in tweet:
                continue
            if tweet['lang'] == options.lang:
                tokens = tokenizer.tokenize(Utils.tweet_text(tweet))
                o.write('%s\n' % json.dumps(tokens))
        
main()    
