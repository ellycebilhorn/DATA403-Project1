import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import seaborn as sn

def make_lower(s):
    return s.lower()

alc_df = pd.read_csv('alc_dataset.csv').drop(columns='Unnamed: 0')
alc_df["Category Name"] = alc_df["Category Name"].apply(make_lower)

print(alc_df)
category_grouped = alc_df.groupby(['Category Name'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
print(category_grouped)
print(category_grouped.sort_values(by=['Bottles Sold'], ascending=False).head(10))