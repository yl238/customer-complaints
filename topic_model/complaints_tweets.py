import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
import argparse
import logging
import tempfile


# custome libraries
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from topic_model.lda_utils import gen_model_data, build_lda_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/to_nationwide.csv')
    parser.add_argument('--n_features', type=int, default=5000)
    parser.add_argument('--n_topics', type=int, default=15)
    parser.add_argument('--output_path', type=str, default='lda_model.pkl')

    args = parser.parse_args()

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    file = args.input_path
    logger.info('Reading tweet file from {}...'.format(file))
    all_tweets = pd.read_csv(file)

    username = 'asknationwide'
    valid_cols = ['id', 'conversation_id', 'created_at', 'user_id', 'tweet']
    logger.info('Remove tweets from {}, clean and lemmatize'.format(username))
    valid_df = gen_model_data(all_tweets, username, valid_cols)


    n_topics = args.n_topics
    n_vocab_features = args.n_features
    
    text = valid_df['lemmatized']
    logger.info('Building LDA model with {} topics and {} features'.format(
        n_topics, n_vocab_features
    ))
    vectorizer, lda_model = build_lda_model(text, n_vocab_features, n_topics)
    
    logger.info("Computing model perplexity...")
    X = vectorizer.transform(text)
    logger.info('Model has perplexity {}'.format(lda_model.perplexity(X)))

    lda_topics = lda_model.transform(X)
    
    logger.info('Saving model, vectorizer and original data...')
    data = {'lda_model': lda_model, 'vectorizer': vectorizer, 
            'orig_df': valid_df, 'topics':lda_topics, 'vectorized':X}
    
    output_path = args.output_path
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    
    