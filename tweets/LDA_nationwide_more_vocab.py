import re
import numpy as np
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import gensim
import gensim.utils
from gensim.utils import simple_preprocess
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
            if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def add_lemmatized_data(df):
    """Remove stop words and lemmatize"""
    data_words = list(sent_to_words(df.cleaned.values))
    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    df['lemmatized'] = [' '.join(tokens) for tokens in data_lemmatized]
    return df
    
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ",".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def clean_doc(doc):
    pattern = re.compile('@\S+|http\S+|pic.\S+|www\S+|\w*\d\w*|\s*[^\w\s]\S*')
    return pattern.sub('', doc).strip()
    
def gen_model_data(all_tweets, username, valid_cols):
    """Remove all Nationwide and short tweets, remove tweet related extra symbols"""
    length_filter = ((all_tweets.username != username) & (all_tweets.tweet.str.len()>14))|(all_tweets.username == username)
    
    model_df = all_tweets[length_filter][valid_cols]
    model_df['cleaned'] = model_df['tweet'].apply(clean_doc).apply(lambda x: None if len(x) < 10 else x)
    valid_df = model_df.dropna(subset=['cleaned'])
    return valid_df

if __name__ == '__main__':
    file = 'data/to_nationwide.csv'
    all_tweets = pd.read_csv(file)

    username = 'asknationwide'
    valid_cols = ['id', 'conversation_id', 'created_at', 'user_id', 'tweet']
    
    valid_df = gen_model_data(all_tweets)
    valid_df = add_lemmatized_data(valid_df)
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    vectorizer.fit(valid_df['lemmatized'])
    X = vectorizer.transform(valid_df['lemmatized'])
    
    n_topics = 15
    lda_model = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=0.1, 
                                              topic_word_prior=0.1, random_state=42, n_jobs=-1)
    lda_model.fit(X)
    print("Model perplexity: {}".format(lda_model.perplexity(X)))
    lda_topics = lda_model.transform(X)
    
    data = {'lda_model': lda_model, 'vectorizer': vectorizer, 'orig_df': valid_df, 'topics':lda_topics, 'vectorized':X}
    with open('lda_5000vocab_15_topics.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    
    