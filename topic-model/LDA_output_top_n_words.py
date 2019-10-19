import numpy as np
import pickle
import time
import pandas as pd


def gen_topic_map(map_file):
    """
    Read topic to description map file (an Excel file) and convert to dictionary
    
    Parameters
    ----------
    map_file: Excel file with two columns, first is the topic ID and second the 
              approximate topic names (derived by the modeller).
              
    Returns
    -------
    Topic ID to name mapping, dict
    """
    topic_map = pd.read_excel(map_file).dropna()[['ID', 'Summarized Topic Name']]
    topic_map['ID'] = topic_map['ID'].astype(int)
    topic_map = topic_map.set_index('ID')
    topic_map = list(topic_map.to_dict().values())[0]
    return topic_map


def extract_complaint_top_words(lda_model, vectorizer, n_top_words=20, topic_map=None):
    """
    Generate the top N words using models trained on US data.
    Both model and vectorized text are trained original CFPB text, and reloaded from pickled file.
    
    NOTE: This works with a pre-trained LDA model, so if you train it again, the clusters will re-order
    and the topic_map will not be correct!!!
    
    Parameters
    ----------
    lda_model - scikit-learn LDA model: trained on the CFPB complaints
    vectorizer - scikit-learn Countvectorizer transformed data: applied on the CFPB complaints.
    n_top_words - int: Number of words to display (default = 20)
    topic_map - dict: mapping of topic idx to name
    
    Returns
    -------
    Pandas DataFrame - Topics with the top_n_words.    
    """
    feature_names = vectorizer.get_feature_names()
    topic_top_words = {}
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        
        if topic_map:
            top_words.append(topic_map[topic_idx])
        
        topic_top_words[topic_idx] = top_words
       
    df = pd.DataFrame.from_records(topic_top_words).T
    df.rename(columns={20: 'Topic name'}, inplace=True)
    return df.set_index('Topic name')


if __name__ == '__main__':
    # Read pickled model and vectorized data
    with open('input/lda_45_topics.pkl', 'rb') as f:
        data = pickle.load(f)
    lda_model = data['model']
    vectorizer = data['vectorizer']
    
    # Read topic to name file
    topic_map = gen_topic_map('input/topics_matching.xlsx')

    # Get the top n words for each topic and save as CSV
    top_words_df = extract_complaint_top_words(lda_model, vectorizer, n_top_words=20, topic_map=topic_map)
    top_words_df.to_csv('output/LDA_top_words.csv')
