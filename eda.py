import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime
import numpy as np
import pandas as pd
import seaborn as sn

figure(figsize=(8, 6), dpi=80)

'''
Exploratory Data Analysis
'''
def get_week(s):
    split_date = s.split('/')
    return datetime.date(int(split_date[2]), int(split_date[0]), int(split_date[1])).isocalendar()[1]

def make_lower(s):
    return s.lower()

alc_df = pd.read_csv('alc_dataset.csv').drop(columns='Unnamed: 0')
grouped_df = pd.read_csv('grouped_df.csv').drop(columns='Unnamed: 0')

corr_df = grouped_df[['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'State Bottle Retail', 'Bottles Sold', 'Sale (Dollars)', 'Volume Sold (Liters)']]
corr_df2 = grouped_df.drop(columns=['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'State Bottle Retail', 'Bottles Sold', 'Sale (Dollars)', 'Volume Sold (Liters)'])
corrMatrix = corr_df.corr()
sn.heatmap(corrMatrix, annot=True)
#plt.show()
#print(corrMatrix)
# print(alc_df.columns)
# print(grouped_df.columns)
# print(alc_df)
# print(grouped_df)

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

x_axis = year_grouped['Year']
y_axis1 = year_grouped['Bottles Sold']
y_axis2 = year_grouped['Volume Sold (Liters)']
y_axis3 = year_grouped['Sale (Dollars)']
plt.plot(x_axis, y_axis1)
plt.xlabel('Year')
plt.ylabel('Bottles Sold')
plt.show()

plt.plot(x_axis, y_axis2)
plt.xlabel('Year')
plt.ylabel('Volume Sold (Liters)')
plt.show()

plt.plot(x_axis, y_axis3)
plt.xlabel('Year')
plt.ylabel('Sale (Dollars)')
plt.show()

# HOLIDAY ANALYSIS
county_grouped = alc_df.groupby(['County'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
#print(county_grouped.sort_values(by=['Bottles Sold'], ascending=False))

polk_2019_df = alc_df[(alc_df['Year'] == 2019) & (alc_df['County'] == 'POLK')]
polk_2020_df = alc_df[(alc_df['Year'] == 2020) & (alc_df['County'] == 'POLK')]
polk_2019_df["Week"] = polk_2019_df["Date"].apply(get_week)
polk_2020_df["Week"] = polk_2020_df["Date"].apply(get_week)
#print(polk_2020_df)
#print(polk_2020_df.dtypes)
week_grouped_2019 = polk_2019_df.groupby(['Week'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
week_grouped_2020 = polk_2020_df.groupby(['Week'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()

print(week_grouped_2019)
print(week_grouped_2020)

print(week_grouped_2019["Bottles Sold"].mean())
print(week_grouped_2020["Bottles Sold"].mean())

# 2019 plots
x_axis = week_grouped_2019['Week']
y_axis1 = week_grouped_2019['Bottles Sold']
y_axis2 = week_grouped_2019['Volume Sold (Liters)']
y_axis3 = week_grouped_2019['Sale (Dollars)']
plt.plot(x_axis, y_axis1)
plt.xlabel('Week')
plt.ylabel('Bottles Sold')
plt.title("Polk County 2019")
plt.show()

plt.plot(x_axis, y_axis2)
plt.xlabel('Week')
plt.ylabel('Volume Sold (Liters)')
plt.title("Polk County 2019")
plt.show()

plt.plot(x_axis, y_axis3)
plt.xlabel('Week')
plt.ylabel('Sale (Dollars)')
plt.title("Polk County 2019")
plt.show()

# 2020 plots
x_axis = week_grouped_2020['Week']
y_axis1 = week_grouped_2020['Bottles Sold']
y_axis2 = week_grouped_2020['Volume Sold (Liters)']
y_axis3 = week_grouped_2020['Sale (Dollars)']
plt.plot(x_axis, y_axis1)
plt.xlabel('Weeks (2020)')
plt.ylabel('Bottles Sold')
plt.title("Polk County 2020")
plt.show()

plt.plot(x_axis, y_axis2)
plt.xlabel('Weeks (2020)')
plt.ylabel('Volume Sold (Liters)')
plt.title("Polk County 2020")
plt.show()

plt.plot(x_axis, y_axis3)
plt.xlabel('Weeks (2020)')
plt.ylabel('Sale (Dollars)')
plt.title("Polk County 2020")
plt.show()

# MOST POPULAR ALCOHOL ANALYSIS
alc_df["Category Name"] = alc_df["Category Name"].apply(make_lower)

category_grouped = alc_df.groupby(['Category Name'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
print(category_grouped.sort_values(by=['Bottles Sold'], ascending=False).head(10))