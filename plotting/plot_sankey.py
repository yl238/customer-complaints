import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import urllib, json

def plot_sankey(df, colors=None): 
    """
    Plot Sankey diagram to show different contribution of different companies to products.
    Might be a better way to generalise this.
    """
    valid_companies = [k for k, v in dict(df['Company'].value_counts()).items() if v > 10000]
    valid_types = [k for k, v in dict(df['Product'].value_counts()).items() if v > 10000]
    
    n_companies = len(valid_companies)
    company_map = {name: idx for idx, name in enumerate(valid_companies)}
    service_map = {name: idx+n_companies for idx, name in enumerate(valid_types)}
    
    valid_df = df[(df.Company.isin(valid_companies)) & (df.Product.isin(valid_types))][['Product', 'Company']]
    valid_df['count'] = 1
    valid_df = valid_df.groupby(['Product', 'Company']).count().unstack().fillna(0)
    
    final_df = valid_df.unstack().unstack().unstack()
    source = [company_map[val[1]] for val in final_df.columns]
    target = [service_map[val[0]] for val in final_df.columns]
    value = list(final_df.values[0])
    
    n_data = len(value)

    node_labels = valid_companies + valid_types
    if colors:
        node_colors = colors[:n_data]
    else:
        node_colors = ['black']*n_data
    
    
    fig = go.Figure(data=[go.Sankey(
        valueformat='d',
        node = dict(
            pad=15,
            thickness=15,
            line=dict(color='black', width=0.5),
            label=node_labels,
            color=node_colors),
        link = dict(
            source=source,
            target=target,
            value=value,
            label=['']*len(node_labels))
    )])

    fig.update_layout(
        font=dict(size=12),
        width=1500,
        height=1200)
    fig.show()
    return fig


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    colors = data['data'][0]['node']['color']

    df = pd.read_csv('../data/Consumer_Complaints.csv')

    fig = plot_sankey(df, colors)
    plotly.offline.plot(fig, filename='sankey.html')