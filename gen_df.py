import numpy as np
import pandas as pd
from pathlib import Path  

'''
File that generates the grouped dataframe
'''

def get_month(string):
    return string.split("/")[0]

def make_lower(s):
    return s.lower()

alc_df = pd.read_csv('alc_dataset.csv', nrows=3000000)
#alc_df = pd.read_csv('alc_dataset.csv')
alc_df["Month"] = alc_df["Date"].apply(get_month)
alc_df["Category Name"] = alc_df["Category Name"].apply(make_lower)
alc_df["Vendor Name"] = alc_df["Vendor Name"].apply(make_lower)
alc_df["Store Name"] = alc_df["Store Name"].apply(make_lower)
alc_df.drop(columns = ["Year", "Date", "Unnamed: 0"], inplace = True)

grouped_df = pd.get_dummies(alc_df, columns=['Store Name', 'Category Name', 'Vendor Name']).groupby(['Month', 'County', 'Population']).sum().reset_index()

#grouped_df = alc_df.groupby(['Month', 'County', 'Population']).sum().reset_index()

grouped_df = pd.get_dummies(grouped_df, columns=['Month', 'County'])

filepath = Path('grouped_df.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
grouped_df.to_csv(filepath)