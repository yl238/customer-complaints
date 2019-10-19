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
from text_summarization.text_summarize_textrank import gen_sent_vectors,\
                compute_similarity_matrix, rank_sentences, gen_top_n_sentences


if __name__ == '__main__':
    df = pd.read_csv('../data/product_merged.csv', 
        usecols=['Complaint ID', 'Product', 'Issue', 'Consumer complaint narrative'])
    
    subset = df[df.Product == 'Mortgage']
    
    sentences = tokenize_sentences(subset['Consumer complaint narrative'])
    
    clean_sentences = clean_and_tokenize(sentences)
    
    # Extract word vectors
    word_embeddings = {}
    f = open('../data/embeddings/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    
    data = {'sentences': sentences, 'clean_sentences': clean_sentences, 
            'embeddings': word_embeddings, 
            'complaint_id': subset['Complaint ID'].values}

    with open('results/mortgage_sentences.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    summarized = [gen_top_n_sentences(sentences[idx], 
                                      clean_sentences[idx], 
                                      word_embeddings) for idx in range(len(subset))]
    
    cleaned_summarized = [' '.join(clean_sentences[idx]) for idx in range(len(subset))]
    subset['summarized'] = summarized
    subset['cleaned_summarized'] = cleaned_summarized
    
    subset.to_csv('results/mortgage_with_summaries.csv', index=False)