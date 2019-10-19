# Text preprocessing
Scripts to relabel and clean text

- `relabel_complaint_products.py` - Relabel the `Product` column of US complaints datasets so they group together better (some of the original products have overlapping subproducts)
- `clean_and_tokenize.py` - Clean US complaints dataset of company names and anonymized characters (XXX), remove stopwords and lemmatize using spaCy
- `gensim_preprocess.py` - As above, but preprocessed using the `gensim` package and `nltk` stopwords. More specific to tweets 