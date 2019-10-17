#-----
# separate script: process.py
# process the complaints

import pickle
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import tqdm

storage = defaultdict(dict)

companies = {'www.nationwide.co.uk':60,'www.hsbc.co.uk':100,
             'lloydsbank.com':25,'www.natwest.com':65,
             'www.barclays.co.uk':80,'tsb.co.uk':45,
             'www.halifax.co.uk':50,'www.rbs.co.uk':16}


for company, pages_to_get in companies.items():
    print('Fetching complaints for:',company)
    for page_number in tqdm.tqdm(range(1,pages_to_get+1)):
        page = pickle.load(open('./'+company+'/page_'+str(page_number),'rb'))

        soup = BeautifulSoup(page, 'html.parser')
        
        articles = soup.findAll("article", {"class": "review"})
        complaints = soup.findAll("p", {"class": "review-content__text"})
        scores = soup.findAll("div", {"class": "star-rating star-rating--medium"})
        review_title = soup.findAll('h2',{'class':'review-content__title'})
        

        for i in range(0,len(articles)):
            complaint_id = articles[i]['id']
            print(complaint_id)
            storage[complaint_id]['company'] = company
            storage[complaint_id]['date_time'] = articles[i].text.split('publishedDate":')[1].split(",")[0][1:-1]
            storage[complaint_id]['review_title'] = review_title[i].text[1:-1]
            storage[complaint_id]['complaint_text'] = complaints[i].text
            storage[complaint_id]['score']=scores[i].img['alt']

dat = pd.DataFrame.from_dict(storage).T
dat['score_num'] = (dat.score.str[0]).astype('int')
dat['complaint_text_cleaned'] = dat.complaint_text.str[2:-2].str.strip()
dat.drop(['score','complaint_text'],axis=1).to_csv('complaints_full.csv')