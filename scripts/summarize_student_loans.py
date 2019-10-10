import re
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import os, sys
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from preprocess.clean_and_tokenize import tokenize_sentences, clean_and_tokenize
from scripts.text_summarize_textrank import gen_sent_vectors, compute_similarity_matrix, rank_sentences, gen_top_n_sentences



if __name__ == '__main__':
    df = pd.read_csv('../data/product_merged.csv', usecols=['Complaint ID', 'Product', 'Issue',
                                                       'Consumer complaint narrative'])
    subset = df[df.Product == 'Student loan']
    
    sentences = tokenize_sentences(subset['Consumer complaint narrative'])
    
    clean_sentences = clean_and_tokenize(sentences)
    
    # Extract word vectors
    word_embeddings = {}
    f = open('../ipynb/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    
    data = {'sentences': sentences, 'clean_sentences': clean_sentences, 'embeddings': word_embeddings, 
            'complaint_id': subset['Complaint ID'].values}
    with open('../output/student_loans_sentences.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    summarized = [gen_top_n_sentences(sentences[idx], 
                                      clean_sentences[idx], 
                                      word_embeddings) for idx in range(len(subset))]
    
    cleaned_summarized = [' '.join(clean_sentences[idx]) for idx in range(len(subset))]
    subset['summarized'] = summarized
    subset['cleaned_summarized'] = cleaned_summarized
    
    subset.to_csv('../data/student_loans_with_summaries.csv', index=False)
    
    
    # LDA
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
    
    X = vectorizer.fit_transform(subset['cleaned_summarized'])
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.model_selection import GridSearchCV

    search_params = {'n_components': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}

    # Init the model
    lda = LatentDirichletAllocation(random_state=42)

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(X)
    
    best_lda_model = model.best_estimator_

    # Model parameters
    print("Best model's parameters: {}".format(model.best_params_))

    # Log likelihood score
    print("Best log-likelihood score: {}".format(model.best_score_))

    # Perplexity
    print("Model perplexity: {}".format(best_lda_model.perplexity(X)))
    
    data = {'model': best_lda_model, 'vectorizer': vectorizer}
    with open('../output/student_loans_lda_model.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)