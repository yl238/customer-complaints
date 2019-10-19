from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from preprocess.gensim_preprocess import clean_and_lemmatize


def gen_model_data(all_tweets, username, valid_cols):
    """
    Filter tweets for user and length, clean and lemmatize.
    Raw tweet has to be > 14 characters long.
    Lemmatized text has to be >= 10 characters long.
    """
    length_filter = ((all_tweets.username != username) & 
        (all_tweets.tweet.str.len()>14))|(all_tweets.username == username)
    
    model_df = all_tweets[length_filter][valid_cols]

    model_df['lemmatized'] = clean_and_lemmatize(model_df['tweets'])
    model_df['lemmatized'] = model_df['lemmatized'].apply(lambda x: 
            None if len(x) < 10 else x)
    valid_df = model_df.dropna(subset=['lemmatized'])
    return valid_df


def build_lda_model(text, n_vocab_features=1000, n_topics=15):
    """
    Build LDA topic model with changeable number of topics and 
    number of words in vocab.

    We've deliberately kept the doc_topic_prior and topic_word_prior
    to be constant since it apparently improves performance.
    """
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=n_vocab_features)
    vectorizer.fit(text)
    X = vectorizer.transform(text)
    
    lda_model = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=0.1, 
                                              topic_word_prior=0.1, random_state=42, 
                                              n_jobs=-1)
    lda_model.fit(X)
    return vectorizer, lda_model


def print_top_words(model, feature_names, n_top_words):
    """Print words in each topic with the highest probability."""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ",".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


