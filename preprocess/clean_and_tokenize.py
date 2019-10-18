import re
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


def tokenize_sentences(texts):
    """Use NLTK to split the complaint text into sentences. 
    Note that we read in the all the complaint texts as a list."""
    special = re.compile(r'[\r|\n|\n\r]+') # Remove all the newline characters
    sentences = [sent_tokenize(special.sub(' ', doc)) for doc in texts] 

    return sentences

def decontract(doc):
        # specific
        doc = re.sub(r'’', "'", doc)
        doc = re.sub(r"won\'t", "will not", doc)
        doc = re.sub(r"can\'t", "can not", doc)
        doc = re.sub(r"isn\'t", "is not", doc)
        doc = re.sub(r"tbc", "to be confirmed", doc)
        doc = re.sub(r"t&c", "terms and conditions", doc)
        
        # general
        doc = re.sub(r"weren\'t", "were not", doc)
        doc = re.sub(r"n\'t", " not", doc)
        doc = re.sub(r"\'re", " are", doc)
        doc = re.sub(r"\'s", " is", doc)
        doc = re.sub(r"\'d", " would", doc)
        doc = re.sub(r"\'ll", " will", doc)
        doc = re.sub(r"\'t", " not", doc)
        doc = re.sub(r"\'ve", " have", doc)
        doc = re.sub(r"\'m", " am", doc)
        return doc

def clean_text(doc):
    decontracted = decontract(doc.lower())
    special = re.compile(r'#\S+|http\S+|www\S+|xx+|pic\.\S+|[^a-zA-Z ]+')
    remove_list = ["american express", "wells fargo", "fargo", "bank of america", 
                   "bank america", "ocwen", "equifax", "morgan", "chase", "citibank", 
                   "navient", "nationstar", "capital", "citi", "amex", "capital one", "synchrony", "costco",
                   "hsbc", "derbyshire", "hey", "oh", "gone", 'transunion'
                   "asknationwide", "nationwide", "twitter", "hi", "yes", "yep", "have",
                   "going", "be", "sorry", "hello", "thanks", "thank", "okay", "ok", "get", "to", "no", "not",]
    remove = '|'.join(remove_list)
    pattern = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    doc = pattern.sub('', special.sub(' ', decontracted))
        
    return doc


def clean_and_tokenize_one(doc):
    doc = clean_text(doc)
    doc = nlp(doc, disable=['tagger', 'parser'])
    doc = ' '.join(' '.join(token.lemma_ if not token.is_stop else '' for token in doc).split())    
    return doc
    
    
def clean_and_tokenize(sentences):
    """Remove all weblinks, the XXX anonymiser and any non-alphabetic characters (except spaces)"""
    special = re.compile(r'http\S+|www\S+|[^a-zA-Z ]+|xx+')
    docs = [[' '.join(special.sub('', doc.lower()).split()) for doc in decontract(s)] for s in sentences] 

    clean_sentences = []
    for s in docs:
        tokenized = []
        for doc in nlp.pipe(s, disable=['parser', 'ner']):
            tokenized.append(" ".join(" ".join(token.lemma_ 
                                   if token.lemma_ not in ['-PRON-'] and not token.is_stop else ''
                                     for token in doc).split())) 
        clean_sentences.append(tokenized)
    return clean_sentences


if __name__ == '__main__':
    file = '../data/product_merged.csv'
    df = pd.read_csv(file, low_memory=False)
    
    sentences = tokenize_sentences(df['Consumer complaint narrative'])
    
    clean_sentence_tokenized = clean_and_tokenize(sentences)
    
    df['clean_tokenized'] = clean_sentence_tokenized
    usecols = ['Complaint ID', 'Product', 'Issue', 'Consumer complaint narrative',
              'clean_tokenized']
    df[usecols].to_csv('../data/with_tokenized.csv', index=False)