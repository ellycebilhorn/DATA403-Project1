import numpy as np
import pandas as pd
import requests
import json
from pathlib import Path  

'''
File that generates the main dataframe
'''
def return_name(string):
  return " ".join(string.split(" ")[:-1])

def return_year(string, delimiter, index):
  return string.split(delimiter)[index]

# Combined data cleaning
def remove_last(input):
    store = input.strip().split(" ")
    if(np.char.isnumeric(store[-1])):
        return ' '.join(store[:-1])
    return ' '.join(store)

def get_month(string):
    return string.split("/")[0]

def make_lower(s):
    return s.lower()


url = "/datasets/Iowa_Liquor_Sales.csv"
alc_df = pd.read_csv(url).dropna()

pop_df= pd.read_csv("https://data.iowa.gov/resource/qtnr-zsrc.csv")
pop_df["County"] = pop_df["geographicname"].apply(return_name).str.upper()
pop_df["Year"] = pop_df["year"].apply(return_year, args=("-", 0))
alc_df["Year"] = alc_df["Date"].apply(return_year, args=("/", 2))
alc_df["County"] = alc_df["County"].str.upper()

df = pd.read_html("https://www.iowa-demographics.com/counties_by_population")[0]
df["Year"] = "2021"
df["County"] = df["County"].apply(return_name).str.upper()
df["Population"] = df["Population"]
df = df[["County", "Population", "Year"]]
pop_df["Population"] = pop_df["population"]
pop_df = pop_df[["County", "Population", "Year"]]
pop_df = pd.concat([pop_df, df])

combined = alc_df.merge(pop_df, how = "left")
combined.drop(columns = ["Address", "City", "Zip Code", "Store Location", "County Number", "Invoice/Item Number", "Store Number", "Category", "Item Description", "Item Number", "Vendor Number", "Pack", "Volume Sold (Gallons)"], inplace = True)
combined = combined.dropna()
combined["Store Name"] = combined["Store Name"].apply(return_year, args=("/", 0)).apply(return_year, args=("#", 0))
combined["Store Name"] = combined["Store Name"].apply(remove_last)
filepath = Path('alc_dataset.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
combined.to_csv(filepath)  