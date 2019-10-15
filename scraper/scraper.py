import asyncio
from pyppeteer import launch
import pickle

async def main(page_num,company_url):
    browser = await launch({"headless": False})
    page = await browser.newPage()
    if page_num>1:
        await page.goto('https://uk.trustpilot.com/review/'+company_url+'?page='+str(page_num))
    else:
        await page.goto('https://uk.trustpilot.com/review/'+company_url)
    await page.waitForSelector('div.review-content__body')

    #okay got that all working, should be able to pull the contents from that page now.
    pagecontent = await page.content()
    await browser.close()

    return pagecontent


import os
# company url to parse and how many pages to grab
companies = {'www.nationwide.co.uk':60,'www.hsbc.co.uk':100,'lloydsbank.com':25,'www.natwest.com':65,'www.barclays.co.uk':80,'tsb.co.uk':45,'www.halifax.co.uk':50,'www.rbs.co.uk':16}

#this sometimes files, so just rerun the script until it is all done.
for company, pages_to_get in companies.items():
    if not os.path.exists('./'+company):
        os.mkdir('./'+company)
    for page_number in range(1,pages_to_get+1):
        if os.path.exists('./'+company+'/page_'+str(page_number)):
            pass
        else:
            output = asyncio.get_event_loop().run_until_complete(main(page_number,company_url=company))
            # parse this with beautiful soup.
            pickle.dump(output,file =open('./'+company+'/page_'+str(page_number),'wb'))



#-----
# separate script: process.py
# process the complaints

import pickle
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import tqdm

storage = defaultdict(dict)

companies = {'www.nationwide.co.uk':60,'www.hsbc.co.uk':100,'lloydsbank.com':25,'www.natwest.com':65,'www.barclays.co.uk':80,'tsb.co.uk':45,'www.halifax.co.uk':50,'www.rbs.co.uk':16}


for company, pages_to_get in companies.items():
    print('Fetching complaints for:',company)
    for page_number in tqdm.tqdm(range(1,pages_to_get+1)):
        page = pickle.load(open('./'+company+'/page_'+str(page_number),'rb'))

        soup = BeautifulSoup(page, 'html.parser')

        articles = soup.findAll("article", {"class": "review"})
        compliants = soup.findAll("p", {"class": "review-content__text"})
        scores = soup.findAll("div", {"class": "star-rating star-rating--medium"})
        review_title = soup.findAll('h2',{'class':'review-content__title'})

        for i in range(0,len(articles)):
            complaint_id = articles[i]['id']
            storage[complaint_id]['company'] = company
            storage[complaint_id]['date_time'] = articles[i].text.split('publishedDate":')[1].split(",")[0][1:-1]
            storage[complaint_id]['review_title'] = review_title[i].text[1:-1]
            storage[complaint_id]['compliant_text'] = compliants[i].text
            storage[complaint_id]['score']=scores[i].img['alt']

dat = pd.DataFrame.from_dict(storage).T
dat['score_num'] = (dat.score.str[0]).astype('int')
dat['compliant_text_cleaned'] = dat.compliant_text.str[2:-2].str.strip()
dat.drop(['score','compliant_text'],axis=1).to_csv('complaints_full.csv')

