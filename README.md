# Customer Complaint Analysis
This directory contains a number of NLP experiments to derive insights from customer complaints data. There are also some dynamic visualization demos.

## Datasets used
1. [US Bureau of Consumer Financial Protection](https://catalog.data.gov/dataset/consumer-complaint-database) Complaints Database. These are complaints about financial products and services. This dataset currently contains about 1,500,000 complaints and is growing.

2. [UK Trustpilot Reviews](https://uk.trustpilot.com/) for a number of financial organisations, including Nationwide, HSBC, Barclays, RBS. These contain free text and also ratings from 1-5.

3. Twitter feeds for the above organizations.

## Experiments performed
1. Customer Segmentation - predict financial product based on complaint texts (US data)
2. Text summarization: use Pagerank to condense long complaints to manageable lengths (US data, but will work on any piece of text)
3. Topic modelling: LDA to cluster complaint topics from the US data and predict topics on UK complaints (US data, Trustpilot, Twitter) 
 


