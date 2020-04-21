import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set(font_scale=1.6, font='Arial')

import plotly.offline as pyo
import plotly.graph_objects as go
import plotly


def plot_proportions(df):
    """
    Plot pie chart showing the proportion of complaint categories.
    """
    product_counts = df['Product'].value_counts().to_dict()

    shortened = ['Credit Reporting', 'Debt Collection', 'Mortgage', 'Credit Card', 'Banking Service', 'Student Loan',
            'Consumer Loan', 'Money Transfer', 'Personal Loan', 'Others']
    labels = list(product_counts.keys())
    values = list(product_counts.values())
   
    fig = {'data' : [{'type' : 'pie',
                    'name' : "Complaint Type",
                    'labels' : labels,
                    'values' : values,
                    'direction' : 'clockwise',
                    'showlegend': False,
                    'text': shortened,
                    'textinfo': 'text+value+percent'
                    }],
        'layout' : {'title' : 'Product Complaint by Type'}}
    fig = go.Figure(fig)
    plotly.offline.plot(fig, filename='proportions.html')


if __name__ == '__main__':
    df = pd.read_csv('../data/with_tokenized.csv')
    plot_proportions(df)