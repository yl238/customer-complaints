import re
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


if __name__ == '__main__':
    file = '../data/product_merged.csv'
    df = pd.read_csv(file, low_memory=False)
    
    special = re.compile(r'http\S+|www\S+|[^a-zA-Z ]+|xx+')
    docs = [' '.join(special.sub('', doc.lower()).split()) \
            for doc in df['Consumer complaint narrative'].values]
    
    df['cleaned_text'] = docs
    tokenized = []
    for doc in nlp.pipe(docs, disable=['tagger', 'parser', 'ner']):
        tokenized.append(" ".join(token.lemma_.lower() for token in doc \
                                  if not token.is_stop and not token.is_space \
                                  and not token.is_punct and not token.like_num))
        
    df['tokenized_text'] = tokenized
    df.to_csv('../data/with_tokenized.csv', index=False)