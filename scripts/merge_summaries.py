"""Concatenate the summary files together."""
import re
import pandas as pd

if __name__ == '__main__':
    text_file = '../output/summary_1.txt'
    text = []
    with open(text_file, 'r') as f:
        for line in f:
            text.append(line)
    text_file = '../output/summary_100000.txt'
    with open(text_file, 'r') as f:
        for line in f:
            text.append(line)
    text_file = '../output/summary_200000.txt'
    with open(text_file, 'r') as f:
        for line in f:
            text.append(line)
    text_file = '../output/summary_300000.txt'
    with open(text_file, 'r') as f:
        for line in f:
            text.append(line)
            
    # Remove the new line tag due to reading from a text file
    no_newlines = [re.sub(r'\n', '', doc) for doc in text]
    summary_df = pd.DataFrame()
    summary_df['summary'] = no_newlines
    summary_df.to_csv('../output/all_summaries_10_sent.csv')
