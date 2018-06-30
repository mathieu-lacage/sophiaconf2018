
import requests
import requests_oauthlib
import json
import optparse
import sys

class NetworkReadError(Exception):
    pass
class NetworkConnectionError(Exception):
    pass
class HttpInternalError(Exception):
    pass
class HttpRateLimitError(Exception):
    pass

def connect(consumer_key, consumer_secret, token, token_secret, keywords):
    auth = requests_oauthlib.OAuth1(consumer_key, consumer_secret, token, token_secret)
    try:
        r = requests.get('https://stream.twitter.com/1.1/statuses/filter.json',
                         params = {'track' : ','.join(keywords)},
                         stream = True,
                         auth = auth)
    except:
        raise NetworkConnectionError()        
    if r.status_code == 420:
        raise HttpRateLimitError()
    if r.status_code == 503:
        raise HttpInternalError()
    if r.status_code != 200:
        print r.text
        raise Exception("Un-expected http error status=%d" % r.status_code)
    return r


def read(r):
    lines = r.iter_lines()
    while True:
        try:
            line = next(lines)
        except requests.exceptions.ConnectionError as e:
            raise NetworkReadError()
        if line:
            try:
                data = json.loads(line)
            except:
                raise Exception("Incorrect data received: %s" % line)
            yield data


def main():
    parser = optparse.OptionParser()
    parser.add_option('--consumer-key', default='OcivAjJVUhZ3L5QEPWR3TEJSO')
    parser.add_option('--consumer-secret', default='dUhh7pDUyMsODrChUkSlrV5JDJNRhtTAvams7gr96osiTmJcnI')
    parser.add_option('--token', default='399983938-kPOVoauPrqmzLNj2VflYGt0eDrDW4FXS0IRbMByd')
    parser.add_option('--token-secret', default='An0Z8L6xFjfLufPlTPMWeXhJ5XxiyI0SttC4e0EwGqghJ')
    parser.add_option('-k', '--keyword', default=[], action='append')
    parser.add_option('--dataset', default='sample')
    options, args = parser.parse_args()

    r = connect(consumer_key = options.consumer_key, 
                consumer_secret= options.consumer_secret, 
                token = options.token, 
                token_secret= options.token_secret,
                keywords = options.keyword)


    with open('%s-tweets.json' % options.dataset, 'a+') as o:
        for d in read(r):
            sys.stdout.write('.')
            sys.stdout.flush()
            o.write('%s\n' % json.dumps(d))
            o.flush()

if __name__ == '__main__':
    main()



