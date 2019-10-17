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


if __name__ == '__main__':
    import os
    # company url to parse and how many pages to grab
    companies = {'www.nationwide.co.uk':60,'www.hsbc.co.uk':100,
                 'lloydsbank.com':25,'www.natwest.com':65,
                 'www.barclays.co.uk':80,'tsb.co.uk':45,
                 'www.halifax.co.uk':50,'www.rbs.co.uk':16}
    
    #this sometimes fails, so just rerun the script until it is all done.
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
