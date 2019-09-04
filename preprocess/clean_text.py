import re
import json
from bs4 import BeautifulSoup
import unicodedata


class TextClean(object):
    def __init__(self, lower=False, min_sentence_length=5):
        self.lower = lower
        self.min_sentence_length = min_sentence_length
       
    @classmethod
    def strip_html_tags(cls, text):
        soup = BeautifulSoup(text, 'html.parser')
        stripped_text = soup.get_text()
        return stripped_text

    @classmethod
    def remove_accented_chars(cls, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    @classmethod
    def expand_contractions(cls, text, contractions_mapping=None):
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_mapping.keys())),
                                        flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contractions_mapping.get(match) \
            if contractions_mapping.get(match) \
            else contractions_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    @classmethod
    def decontract(cls, doc):
        # specific
        doc = re.sub(r"won\'t", "will not", doc)
        doc = re.sub(r"can\'t", "can not", doc)
        doc = re.sub(r"tbc", "to be confirmed", doc)
        # general
        doc = re.sub(r"n\'t", " not", doc)
        doc = re.sub(r"\'re", " are", doc)
        doc = re.sub(r"\'s", " is", doc)
        doc = re.sub(r"\'d", " would", doc)
        doc = re.sub(r"\'ll", " will", doc)
        doc = re.sub(r"\'t", " not", doc)
        doc = re.sub(r"\'ve", " have", doc)
        doc = re.sub(r"\'m", " am", doc)
        return doc

    @classmethod
    def remove_special_characters(cls, text):
        text = re.sub("[^a-zA-Z0-9\s]", ' ', text)
        text = re.sub("XX+",' ', text)
        return text

    @classmethod
    def remove_weblinks(cls, text):
        text = re.sub('http\S+|www\S+', ' ', text)
        return text

    def fit_transform(self, corpus, html_stripping=True, contraction_expansion=True,
                        accented_char_removal=True,
                        special_char_removal=True, contraction_mapping=None,
                        return_indices=True):

        normalized_corpus = []
        valid_idx = []
        # normalize each document in the corpus
        i = 0
        for doc in corpus:
            doc = self.remove_weblinks(doc)
            # strip html
            if html_stripping:
                doc = self.strip_html_tags(doc)
            # remove accented characters
            if accented_char_removal:
                doc = self.remove_accented_chars(doc)
            # expand contractions
            if contraction_expansion:
                if contraction_mapping:
                    doc = self.expand_contractions(doc, contraction_mapping)
                else:
                    doc = self.decontract(doc)
                      
            # Remove extra newlines
            doc = re.sub(r'[\r|\n|\n\r|\n]+', ' ', doc)

            # remove special characters
            if special_char_removal:
                # insert spaces between special characters to isolate them
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1", doc)     
                doc = self.remove_special_characters(doc)
            
            # lowercase the text
            if self.lower:
                doc = doc.lower()
             # remove extra whitespace
            doc = re.sub(' +', ' ', doc)

            # Ensure always return sentence greater than a set number
            # of words    
            if len(doc.split(' ')) > self.min_sentence_length:
                normalized_corpus.append(doc)
                valid_idx.append(i)           
            
            i += 1
        if return_indices:
            return normalized_corpus, valid_idx
        else:
            return normalized_corpus
