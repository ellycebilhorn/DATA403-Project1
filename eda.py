import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
import seaborn as sn

figure(figsize=(8, 6), dpi=80)



alc_df = pd.read_csv('alc_dataset.csv').drop(columns='Unnamed: 0')
grouped_df = pd.read_csv('grouped_df.csv').drop(columns='Unnamed: 0')

corr_df = grouped_df[['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'State Bottle Retail', 'Bottles Sold', 'Sale (Dollars)', 'Volume Sold (Liters)']]
corr_df2 = grouped_df.drop(columns=['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'State Bottle Retail', 'Bottles Sold', 'Sale (Dollars)', 'Volume Sold (Liters)'])
corrMatrix = corr_df.corr()
sn.heatmap(corrMatrix, annot=True)
#plt.show()
#print(corrMatrix)
print(alc_df.columns)
print(grouped_df.columns)

print(alc_df)
print(grouped_df)

corrMatrix2 = corr_df2.corr()
print(corrMatrix2)
print(len(corrMatrix2))
sn.heatmap(corrMatrix2, annot=True)
year_grouped = alc_df.groupby(['Year'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
print(year_grouped)

for year in range(2012, 2022, 1):
    test_df = alc_df.loc[alc_df['Year'] == year]
    print("Year", year, "number of observations:", test_df.shape[0])

#test_df = alc_df.loc[alc_df['Year'] == 2017]
# test_df2 = alc_df.loc[alc_df['Year'] == 2018]
# print(test_df)
# print(test_df2)

# x_axis = year_grouped['Year']
# y_axis1 = year_grouped['Bottles Sold']
# y_axis2 = year_grouped['Volume Sold (Liters)']
# y_axis3 = year_grouped['Sale (Dollars)']
# plt.plot(x_axis, y_axis1)
# plt.xlabel('Year')
# plt.ylabel('Bottles Sold')
# plt.show()

# plt.plot(x_axis, y_axis2)
# plt.xlabel('Year')
# plt.ylabel('Volume Sold (Liters)')
# plt.show()

# plt.plot(x_axis, y_axis3)
# plt.xlabel('Year')
# plt.ylabel('Sale (Dollars)')
# plt.show()

