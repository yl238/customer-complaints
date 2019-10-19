# Analysis and Predictions of LDA
This directory consists of scripts to 
1. Analyse a pretrained Latent Dirichlet Allocation model.
2. predict the topics of complaints from Trustpilot.

### Description of the files
#### Input
- `lda_45_topics.pkl` The LDA model is trained on the [CFPB (Bureau of Consumer Financial Protection) data](https://catalog.data.gov/dataset/consumer-complaint-database) with 45 topics. Also contains the word vectors obtained using scikit-learn's `Countvectorizer` with `ngrams=(1,2)`
- `topics_matching.xlsx` LDA topic indices matched with the hand-labeled topic names.
- `Nationwide_complaints.csv` Complaints about Nationwide Building Society on Trustpilot.

#### Scripts
- `LDA_output_n_top_words.py` - Get the top N words from each topic and save to csv file.
- `LDA_identify_topics.py` - Find the topics of a customer complaint.

