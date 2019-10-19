import pickle
import re
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def gen_sent_vectors(clean_sentences, word_embeddings):
    """
    Generate sentence vectors using word embeddings.
    """
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors


def compute_similarity_matrix(one_complaint, sentence_vectors):
    """
    Compute similarity matrix between sentences within one complaint.
    """
    sim_mat = np.zeros([len(one_complaint), len(one_complaint)])
    for i in range(len(one_complaint)):
        for j in range(len(one_complaint)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100),
                                                 sentence_vectors[j].reshape(1, 100))[0, 0]
    return sim_mat


def rank_sentences(sim_mat, one_complaint, maxlen):
    """Rank sentences using their PageRank score"""
    nx_graph = nx.from_numpy_array(sim_mat)
    try: # Note that PageRank is not guaranteed to converge, hence return the same sentences
        scores = nx.pagerank(nx_graph, max_iter=100)
    except Exception:
        scores = np.zeros(len(one_complaint))
    
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(one_complaint)), reverse=True)
    return ranked_sentences


def gen_top_n_sentences(one_complaint, clean_sentences, word_embeddings, maxlen=5):
    """
    Get the top n most relevant sentences of each of the complaints.
    This is done by using the TextRank algorithm, in the same spirit
    as the PageRank algorithm, except the nodes of the graph are 
    sentences instead of webpages.
    """
    if len(one_complaint) < maxlen:
        condensed = ' '.join(one_complaint)
    else:
        sentence_vectors = gen_sent_vectors(clean_sentences, word_embeddings)
   
        sim_mat = compute_similarity_matrix(one_complaint, sentence_vectors)
        ranked_sentences = rank_sentences(sim_mat, one_complaint, maxlen)
        condensed_sent = [sent[1] for i, sent in enumerate(ranked_sentences) if i < maxlen]
        
        condensed = ' '.join(re.sub('xx\\S+|XX\\S+', ' ', ' '.join(condensed_sent)).split())
    return condensed

if __name__ == '__main__':
    with open('../output/sentences.pickle', 'rb') as f:
        data = pickle.load(f)

    sentences = data['sentences']
    clean_sentences = data['clean_sentences']
    word_embeddings = data['embeddings']

    with open('../output/summary_300000.txt', 'w') as f:
        for idx in range(300000, len(sentences)):
            if idx % 200 == 0:
                print(idx)
            summary = gen_top_n_sentences(sentences[idx], clean_sentences[idx], maxlen=10)
            f.write('{}\n'.format(summary))