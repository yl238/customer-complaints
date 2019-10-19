import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, LSTM, SpatialDropout1D, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

def generate_debts_data_only(df):
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
    
    valid_types = ['DNO', 'CT', 'WN', 'FS', 'DV', 'TNA']
    model_df = model_df[model_df['target'].isin(valid_types)]
    return model_df 

if __name__ == '__main__':
    file = '../data/with_tokenized.csv'
    df = pd.read_csv(file)

    model_df = generate_debts_data_only(df)
    targets = sorted(model_df['target'].unique())    

    MAX_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250

    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_WORDS, 
                    filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(model_df['tokenized_text'].values)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    X = tokenizer.texts_to_sequences(model_df['tokenized_text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor: ', X.shape)

    y = pd.get_dummies(model_df['target']).values
    print('Shape of label tensor: ', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                            test_size = 0.10, random_state = 42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    # Build Keras model
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 5
    batch_size = 64

    history = model.fit(X_train, y_train, 
                    epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', 
                    patience=3, min_delta=0.0001)])

    pred = model.predict(X_test)
    y_true = [targets[i] for i in np.argmax(y_test, axis=1)]
    y_pred = [targets[i] for i in np.argmax(pred, axis=1)]
    print(classification_report(y_true, y_pred))