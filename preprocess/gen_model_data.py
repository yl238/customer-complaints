import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

from preprocess import TextClean


def text_preprocess(text):
    """Use spaCy to remove stopwords and lemmatize
    """
    return " ".join(token.lemma_ for token in nlp(text)
                    if not token.is_stop)        


def gen_model_data(df, subset=None, features=None, lowercase=True, min_sentence_length=5):
    """
    Drop rows with NaNs in 'Product', 'Issue', 'Date' 
    and 'Consumer complaint narrative' columns.
    
    Create a 'clean_text' column with the 
    """
    df = df.dropna(subset=subset)
    if features:
        df = df[features]
        
    cleaner = TextClean(lower=lowercase, min_sentence_length=min_sentence_length)
    text = df['Consumer complaint narrative'].values
    cleaned_text, idx = cleaner.fit_transform(text, return_indices=True)
    df = df.iloc[idx, :]
    df['cleaned_text'] = cleaned_text
    df['text'] = df['cleaned_text'].apply(text_preprocess)

    return df


if __name__ == '__main__':
    complaints_file = '../data/Consumer_Complaints.csv'
    df = pd.read_csv(complaints_file, low_memory=False)

    features = ['Date received', 'Product', 'Issue', 
       'Consumer complaint narrative', 'Company public response', 'Company',
       'Submitted via', 'Date sent to company', 'Company response to consumer',
       'Timely response?', 'Consumer disputed?', 'Complaint ID']

    nonan_subset = ['Product', 'Issue', 'Date sent to company', \
        'Consumer complaint narrative']

    valid_df = gen_model_data(df, subset=nonan_subset, features=features)
    valid_df.to_csv('../data/cleaned_text.csv', index=False)