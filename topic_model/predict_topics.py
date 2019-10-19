import pandas as pd
import re
import numpy as np
import pickle
import time
import pandas as pd
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from preprocess.clean_and_tokenize import clean_and_tokenize_one

def gen_topic_map(map_file):
    """Read topic to description map file as an Excel file and convert to dictionary
    """
    topic_map = pd.read_excel(map_file).dropna()[['ID', 'Summarized Topic Name']]
    topic_map['ID'] = topic_map['ID'].astype(int)
    topic_map = topic_map.set_index('ID')
    topic_map = list(topic_map.to_dict().values())[0]
    return topic_map


def predict_complaint_topics(lda_model, vectorizer, complaints_df, 
                               text_field = 'compliant_text_cleaned',
                               n_topics=45, topic_map=None,
                               top_n=5):
    """
    Predict the top_n topics with probability.
    
    Parameters
    ----------
    lda_model - Latent Dirichlet Model trained on US data
    vectorizer - CountVectorizer trained on US data
    complaint_df - Pandas Dataframe: all complaint info as a Dataframe
    text_field - string: the field name containing complaint text
    
    n_topics - int: number of topics in the US data
    topic_map - dict: topic to description map
    top_n - int: top N topics to be displayed.
    
    Returns
    -------
    dict: {'original_narrative': text, 
           'topics': {topic_idx', 'topic_name', 'topic_prob'}}
    """
    if topic_map is None:
        print('Need topic index to description mapping!!! Stop now and check!!!')
        return
    
    # Clean complaints text prior to prediction
    complaints_df['cleaned'] = complaints_df[text_field].apply(clean_and_tokenize_one)

    # Apply Countvectorizer transform and then LDA predict
    vectorized = vectorizer.transform(complaints_df['cleaned'])
    topics = lda_model.transform(vectorized)
    
    all_output = []
    for i in range(topics.shape[0]):
        output = dict()
        # We need to output the original narrative for comparison
        output['Original narrative'] = complaints_df[text_field][i]
        topic_indices = np.argsort(topics[i, :])[::-1]
        topic_prob = np.sort(topics[i, :])[::-1]
    
        topics_data = {}
        for importance_count, [idx, prob] in enumerate(list(zip(topic_indices, topic_prob))[:top_n]):
            topics_data[importance_count] = {'topic_name': topic_map[int(idx)],
                                             'topic_prob': prob}
        output['topics'] = topics_data
        all_output.append(output)
    return all_output