# run with
#FLASK_APP=review_view.py FLASK_DEBUG=1 python -m flask run --port=9000

from flask import Flask, redirect, url_for
import flask
import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.cm as cm
import json
from collections import defaultdict
import pickle
import re

# only scored for nationwide right now.
suedat = json.load(open('./data/identified_topics.json','r'))

def colour_by_score(message,scores):
    scaled_score = (scores+1)/2

    output =     "<span style='background-color:rgba"+str(tuple(map(lambda x:int(x*255),cm.RdYlGn(scaled_score))))+";'>"+message+'</span>'
    return output

topics = pd.read_excel('./data/topics_matching.xlsx').dropna()
complaints = pd.read_csv("./data/complaints_full.csv",index_col=0)
model = pickle.load(open('./data/lda_issues_45_topics.pkl','rb'))
vectorizer = pickle.load(open('./data/vectorizer.pkl','rb'))
tokens = pickle.load(open('./data/token_dump.pkl','rb'))


complaints['html'] = None



def token_to_words(vector):
    words_out=[]
    for vec in vector:
        words_out.append(tokens[vec])
    return words_out


def format_scores(message):
    comment = TextBlob(message)
    out_html = ''
    for sentence in comment.sentences:
        #print(str(sentence))
        sent = TextBlob(str(sentence)).sentiment
        #print(sent)
        #sent.polarity
        out_html+= ' '+colour_by_score(str(sentence),sent.polarity)
    return out_html


def format_topics(message,lda_topic):

    for i in range(0,5):
        for word in lda_topic[i]['top_words']:

            #message = message.replace(word,'<span id="topic_'+str(i)+'">'+word+'</span>')
            message = re.sub(r"\b"+word+r"\b",'<div id="topic_'+str(i)+'">'+word+'</div>',message)

    return message


print('Finished.')
app = Flask(__name__)
app.secret_key = "Shhhhhh"

@app.route("/")
def homepage():

    htmldata = complaints[complaints.company == 'www.nationwide.co.uk'][['company', 'date_time', 'review_title', 'compliant_text_cleaned','score_num']].to_html()

    return flask.render_template('index.html',render_data=htmldata)

@app.route('/case/<id>')
def profile(id):
    dat = complaints.loc[id]
    dat['html'] = format_scores(dat['compliant_text_cleaned'])

    caseindex = np.where(complaints.index == id)[0][0]
    topresults = pd.DataFrame.from_dict(suedat[caseindex]['topics'])
    lda_topics = defaultdict(dict)
    for i in range(0,5):
        topic_index = topresults[str(i)]['topic_idx']
        lda_topics[i]['topic_idx'] = topresults[str(i)]['topic_idx']
        lda_topics[i]['topic_name'] = topresults[str(i)]['topic_name']
        lda_topics[i]['topic_prob'] = np.round(topresults[str(i)]['topic_prob'],2)
        lda_topics[i]['width'] = np.round(topresults[str(i)]['topic_prob'],2)*100
        #return back the top 500 words
        lda_topics[i]['top_words'] = token_to_words(np.argsort(model.components_[topic_index])[-250:][::-1])

    dat['lda_html'] = format_topics(dat['compliant_text_cleaned'],lda_topics)

    return flask.render_template('case.html', dat=dat,lda_topics=lda_topics)


if __name__ == "__main__":
    app.run()
