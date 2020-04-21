import re
import numpy as np
import pandas as pd
import pickle
import os, sys
import logging
import argparse

from io import StringIO
from pandas.api.types import is_string_dtype
from sqlalchemy import event, create_engine

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from preprocess.clean_and_tokenize import clean_and_tokenize_one
from topic_model.predict_topics import gen_topic_map, predict_complaint_topics


def get_engine(con_string):
    engine = create_engine(con_string)
    return engine


def clean_rows(df, text_cols):
    for col in text_cols:
        if is_string_dtype(df[col]):
            df[col] = df[col].str.replace('|', '')
    return df


def to_postgres(df, table_name, con, text_cols=None):
    data = StringIO()
    if text_cols is not None:
        df = clean_rows(df, text_cols)
    df.to_csv(data, header=False, index=False, sep='|')
    data.seek(0)
    raw = con.raw_connection()
    curs = raw.cursor()
    try:
        curs.execute('DROP TABLE ' + table_name)
    except:
        raw = con.raw_connection()
        curs = raw.cursor()
        print("{} doesn't exist - CREATING".format(table_name))
    #empty_table = 'CREATE TABLE ' + table_name + ' (\n"idx" TEXT,\n  "company" TEXT,\n  "review_title" TEXT,\n  "date" TIMESTAMP,\n  "review_narrative" TEXT,\n  "topics" JSON\n)'
    empty_table = pd.io.sql.get_schema(df, table_name, con=con)
    empty_table = empty_table.replace('"', '')
    curs.execute(empty_table)
    curs.copy_from(data, table_name, sep='|')
    curs.execute("grant select on {} to grp_dev".format(table_name))
    curs.connection.commit()



if __name__ == '__main__':
    trustpilot_data = 'data/trustpilot/trustpilot_complaints_full.csv'
    output_file = 'results/trustpilot_complaint_topics.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=trustpilot_data)
    parser.add_argument('--output_path', type=str, default=output_file)
    parser.add_argument('--save_to_db', type=bool, default=False)
    args = parser.parse_args()

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    input_file = args.input_path
    output_file = args.output_path
    save_to_db = args.save_to_db
    
    logger.info('Reading file from {}...'.format(input_file))
    df = pd.read_csv(input_file)

    df['date'] = pd.to_datetime(df['date_time'])

    logger.info('Load LDA models...')
    with open('./models/lda_45_topics.pkl', 'rb') as f:
        data = pickle.load(f)
    lda_model = data['model']
    vectorizer = data['vectorizer']

    logger.info('Get complaint topics...')
    text_field = 'compliant_text_cleaned'
    df['cleaned'] = df[text_field].astype(str).apply(clean_and_tokenize_one)
    # Apply Countvectorizer transform and then LDA predict
    vectorized = vectorizer.transform(df['cleaned'])
    topics = lda_model.transform(vectorized)

    # Create topic map table
    topic_map = gen_topic_map('./models/topics_matching.xlsx')
    
    top_n = 5
    all_output = []
    for c in range(topics.shape[0]):
        output = dict()
        # We need to output the original narrative for comparison
        output['idx'] = df['Unnamed: 0'].values[c]
        output['company'] = df['company'].values[c]
        output['review_title'] = df['review_title'].values[c]
        output['date'] = df['date'].values[c]
        output['review_narrative'] = df[text_field].values[c]
        topic_indices = np.argsort(topics[c, :])[::-1]
        topic_prob = np.sort(topics[c, :])[::-1]
        
        topics_data = {}
        for importance_count, [idx, prob] in enumerate(list(zip(topic_indices,\
             topic_prob))[:top_n]):
            topics_data[topic_map[int(idx)]] = prob
        output['topics'] = topics_data
        all_output.append(output)

    output_df = pd.DataFrame.from_records(all_output)

    if save_to_db: 
        logger.info('Save to database...')   
        # Save to postgres database
        with open("/Users/sueliu/cred.secret", 'r') as f:
            con_str = f.read().strip()

        # Save topic names
        topic_map_df = pd.DataFrame(topic_map, index=[0]).T.reset_index()
        topic_map_df.columns=['topic_id', 'topic_name']
        to_postgres(topic_map_df,'prototyping.complaints_topic_names',\
            create_engine(con_str))

        # Save complaints and topic assignment
        pd.io.sql.get_schema(output_df, 'prototyping.complaints_trustpilot')
        to_postgres(output_df, 'prototyping.complaints_trustpilot', get_engine(con_str),\
            text_cols=['review_title', 'review_narrative'])
    else:
        print('Save output to {}'.format(output_file))
        output_df.to_csv(output_file, index=False)