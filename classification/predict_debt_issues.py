"""
Predict different types of debt complaints from free text.
Slightly more ambitious.
"""
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def generate_test_preds(model, test_df):
    X_test, y_test = test_df['tokenized_text'].values, test_df['target']
    X_test_tfidf = tfidf_vect.transform(X_test)
    pred = model.predict(X_test_tfidf)
    
    test_df['pred'] = pred
    pred_proba = model.predict_proba(X_test_tfidf)
    proba_df = pd.DataFrame(pred_proba, columns=['pred_'+t for t in sorted(valid_types)])
    proba_df['max_prob'] = proba_df.max(axis=1)
    proba_df['Complaint ID'] = test_df['Complaint ID'].values

    merged = pd.merge(test_df, proba_df, on='Complaint ID')

    orig_df = pd.read_csv('../data/Consumer_Complaints.csv', 
                      usecols=['Complaint ID', 'Consumer complaint narrative'])
    with_narrative_df = pd.merge(merged, orig_df, on='Complaint ID')
    with_narrative_df.to_csv('../output/debts_predictions.csv', index=False)


if __name__ == '__main__':
    file = '../data/with_tokenized.csv'
    df = pd.read_csv(file)

    debt_df = df[df['Product'] == 'Debt collection']

    model_df = debt_df[['Complaint ID', 'tokenized_text', 'Issue']].dropna()
    model_df = model_df[model_df['tokenized_text'].str.len() >= 10]

    # Abbreviate names of issues to make more manageable
    abbrev_map = {
    'Attempts to collect debt not owed' : 'DNO',
    'Communication tactics': 'CT',
    "Cont'd attempts collect debt not owed": 'CDNO',
    "Disclosure verification of debt": 'DV',
    "False statements or representation": 'FS',
    "Improper contact or sharing of info": 'IC',
    "Taking/threatening an illegal action": 'TIA',
    "Threatened to contact someone or share information improperly": 'IC',
    "Took or threatened to take negative or legal action": 'TNA',
    "Written notification about debt": 'WN'
    }
    model_df['target'] = model_df['Issue'].apply(lambda i: abbrev_map[i])

    # Only predict a subset of the above
    valid_types = ['DNO', 'CT', 'WN', 'FS', 'DV', 'TNA']
    model_df = model_df[model_df['target'].isin(valid_types)]
    targets = sorted(model_df['target'].unique())

    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    X_train, X_val = train_df['tokenized_text'].values, val_df['tokenized_text'].values
    y_train, y_val = train_df['target'].values, val_df['target'].values


    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=1500, 
                                ngram_range=(1, 3))
    tfidf_vect.fit(X_train)
    
    X_train_tfidf = tfidf_vect.transform(X_train)
    X_val_tfidf = tfidf_vect.transform(X_val)

    lr = LogisticRegressionCV(max_iter=500, class_weight='balanced', multi_class='auto',
                       solver='lbfgs', n_jobs=3, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    generate_test_preds(lr, test_df)


    # Random Forest Model
    print('Building a random forest model!')
    from sklearn.ensemble import RandomForestClassifier
    from pprint import pprint

    rf = RandomForestClassifier(random_state=42)
    pprint('Parameters currently in use:')
    pprint(rf.get_params())

    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    pprint(random_grid)