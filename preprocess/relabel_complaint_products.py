"""Relabel some of the Products to maximise data usage"""
import re
import numpy as np
import pandas as pd

def gen_model_data(df, subset=None, features=None):
    """
    Drop rows with NaNs in 'Product', 'Issue', 'Date' 
    and 'Consumer complaint narrative' columns.
    """
    df = df.dropna(subset=subset)
    if features:
        df = df[features]
    return df

def merge_products(df):
    # Merge Payday loan with other loans
    main_type = 'Payday loan, title loan, or personal loan'
    subset_type = 'Payday loan'
    subset = df[df.Product == subset_type]
    subset['Sub-product'] = subset_type
    subset['Product'] = main_type
    main_set = df[df.Product == main_type]
    loans = pd.concat([subset, main_set])
    
    # Credit card
    main_type = 'Credit card or prepaid card'
    subset_type = 'Credit card'
    subset = df[df.Product == subset_type]
    subset['Sub-product'] = subset_type
    subset['Product'] = main_type
    main_set = df[df.Product == main_type]
    credit = pd.concat([subset, main_set])

    subset = df[df.Product == 'Prepaid card']
    credit = pd.concat([subset, credit])
    credit['Product'] = main_type
    
    # Merge vehicle loan with consumer loans
    main_type = 'Consumer Loan'
    subset_type = 'Vehicle loan or lease'
    vehicles = df[df.Product == subset_type]

    loan_types = []
    for old, new in [('Loan', 'Vehicle loan'), ('Lease', 'Vehicle lease'),\
                     ('Title loan', 'Title loan')]:
        idx = vehicles[(vehicles['Sub-product'] == old)]
        idx['Sub-product'] = new
        loan_types.append(idx)
    vehicles = pd.concat(loan_types)

    consumer_loans = df[df.Product == main_type]
    consumer_loans = pd.concat([vehicles, consumer_loans])
    consumer_loans['Product'] = main_type
    
    
    # Merge virtual currency and Money transfer
    main_type = 'Money transfer, virtual currency, or money service'
    subset_type1 = 'Virtual currency'
    subset_type2 = 'Money transfers'
    transfer = df[df.Product == main_type]
    subset1 = df[df.Product == subset_type1]
    subset2 = df[df.Product == subset_type2]

    transfer = pd.concat([subset1, subset2, transfer])
    transfer['Product'] = main_type
    
    # Reporting
    main_type = 'Credit reporting, credit repair services, or other personal consumer reports'
    subset_type = 'Credit reporting'
    subset = df[df.Product == subset_type]
    subset['Sub-product'] = subset_type
    subset['Product'] = main_type
    main_set = df[df.Product == main_type]
    reporting = pd.concat([subset, main_set])
    
    # Banking service
    main_type = 'Bank account or service'
    subset_type = 'Checking or savings account'
    main_set = df[df.Product == main_type]
    subset = df[df.Product == subset_type]

    banking = pd.concat([main_set, subset])
    banking['Product'] = main_type
    
    not_modified = df[df.Product.isin(['Debt collection', \
                                       'Mortgage', 'Student loan','Other financial service'])]
    return pd.concat([not_modified, loans, credit, consumer_loans, transfer, reporting, banking])


def map_abbrev(val):
    return abbrev_map[val]

if __name__ == '__main__':
    complaints_file = '../data/Consumer_Complaints.csv'
    df = pd.read_csv(complaints_file, low_memory=False)
    
    subset = ['Product', 'Issue', 'Consumer complaint narrative']
    df = gen_model_data(df, subset=subset)
    
    merged = merge_products(df)
    abbrev_map = {'Credit reporting, credit repair services, or other personal consumer reports': 'CR',
                'Debt collection': 'DC', 
                'Mortgage': 'MO', 
                'Credit card or prepaid card': 'CC',
                'Bank account or service': 'BS', 
                'Student loan': 'SL', 
                'Consumer Loan': 'CL', 
                'Money transfer, virtual currency, or money service' :'MT', 
                'Payday loan, title loan, or personal loan': 'PL', 
                'Other financial service': 'OT'}
    
    merged['Abbrev'] = merged['Product'].apply(map_abbrev)
    merged.to_csv('../data/product_merged.csv', index=False)
    
    
    