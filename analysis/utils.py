import re
import pandas as pd

def extract_topic_probs(complaints_df):
    """
    Read the output of the topic model and flatten to obtain probabilities of topics.
    """
    pattern = re.compile(r'(www\.|\.co\.uk|\.com)')
    for bank, data in complaints_df.groupby('company'):

        bank_name = pattern.sub('', bank)
        topics_per_company = pd.DataFrame.from_records([eval(t) for t in data['topics'].values])

        data['test'] = np.arange(len(data))
        topics_per_company['test'] = np.arange(len(data))
        df = pd.merge(data, topics_per_company, on='test')
        df.drop(columns=['topics', 'test'], inplace=True)
        #print(df.head())
        output_file = '../ipynb/data/trustpilot_{}.csv'.format(bank_name)
        #print(output_file)
        df.to_csv(output_file, index=False)
    return None


if __name__ == '__main__':
    trustpilot_complaints = '/Users/sueliu/Mudano/customer-complaints/topic_model/results/trustpilot_complaint_topics.csv'
    complaints_df = pd.read_csv(trustpilot_complaints)
    complaints_df['date'] = pd.to_datetime(complaints_df['date'])
    extract_topic_probs(complaints_df)