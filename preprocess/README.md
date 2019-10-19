# Text preprocessing
Scripts to relabel and clean text

- `relabel_complaint_products.py` - Relabel the `Product` column of US complaints datasets so they group together better (some of the original products have overlapping subproducts)
- `tokenize_us_complaints.py` - Simple script to clean and tokenize US complaints data and save to file.
- `clean_and_tokenize.py` - Functions to tokenize, remove US complaints dataset of company names, remove stopwords and lemmatize using spaCy
- `gensim_preprocess.py` - As above, but preprocessed using the `gensim` package and `nltk` stopwords. More specific to tweets 