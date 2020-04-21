"""
Clean and preprocess text using gensim
"""
import re

# NLP libraries
import gensim
import gensim.utils
from gensim.utils import simple_preprocess
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def clean_tweets(texts):
    """
    Remove all non-text characters in a list of tweets
    """
    pattern = re.compile('@\S+|http\S+|pic.\S+|www\S+|\w*\d\w*|\s*[^\w\s]\S*')
    return [pattern.sub('', doc).strip() for doc in texts]


def sent_to_words(sentences):
    """
    Gensim function to decompose sentence into words.
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    """Does what it says, removes nltk stopwords."""
    return [[word for word in simple_preprocess(str(doc)) 
            if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Lemmatization using spacy according to POS"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc 
        if token.pos_ in allowed_postags])
    return texts_out


def clean_and_lemmatize(texts):
    """
    Add to the DataFrame a column with the tweets text cleaned, stopwords 
    removed and lemmatized"""
    cleaned = clean_tweets(texts)
    data_words = list(sent_to_words(cleaned))
    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, 
                allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    lemmatized = [' '.join(tokens) for tokens in data_lemmatized]
    return lemmatized
    