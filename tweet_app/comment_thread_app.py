# run with
#FLASK_APP=comment_thread_app.py FLASK_DEBUG=1 python -m flask run

from flask import Flask, redirect, url_for
import flask
import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.cm as cm
import json
from collections import defaultdict
import pickle


lstm_weights = pickle.load(open('./data/lstm_stored.pkl','rb'))

all_tweets= pd.read_csv('./data/all_tweets.csv')
all_tweets['date'] = pd.to_datetime(all_tweets.date)

length_filter = (((all_tweets.username != 'asknationwide') & (all_tweets.tweet.str.len()>14))|(all_tweets.username == 'asknationwide')) &(all_tweets['date'] > '2019-06-01')

conversation_grouped = all_tweets[(all_tweets.id.isin(lstm_weights.keys())) & length_filter].groupby('conversation_id')
#convoid = np.random.choice(conversation_grouped['id'].count()[conversation_grouped['id'].count() > 3].index)





print('Ready to serve')

def colour_by_scores(message,scores):
    scaled_score = (scores+1)/2

    output = ''
    for x,w in enumerate(message):
        output+="<span style='border-bottom: 4px solid rgba"+str(tuple(map(lambda x:int(x*255),cm.RdYlGn(scaled_score[x]))))+";'>"+w+'</span>'
    return output

#text-decoration: underline;text-decoration-thickness:4px;text-decoration-color:



all_tweets['html'] = ''


print('Finished.')
app = Flask(__name__)


@app.route("/")
def homepage():
    #every time the page loads, show go random page

    convoid = np.random.choice(conversation_grouped['id'].count()[conversation_grouped['id'].count() > 3].index)

    return profile(convoid)

@app.route('/convo/<id>')
def profile(id):
    dat = all_tweets[all_tweets.conversation_id == id].sort_values(['date','time'])
    dat['date'] = pd.to_datetime(dat['date']).dt.strftime('%b %d, %Y')
    for rowd_id, row in dat.iterrows():
        if row['id'] in lstm_weights.keys():
            scores = lstm_weights[row['id']]['char_scores']
            dat.loc[rowd_id,'html'] = colour_by_scores(row['tweet'],scores)
        else:
            dat.loc[rowd_id,'html'] = row['tweet']

    dat_dict = dat.to_dict(orient='records')

    return flask.render_template('case.html', dat=dat_dict)


if __name__ == "__main__":
    app.run()
