import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import seaborn as sn
import datetime

def get_week(s):
    split_date = s.split('/')
    return datetime.date(int(split_date[2]), int(split_date[0]), int(split_date[1])).isocalendar()[1]

alc_df = pd.read_csv('alc_dataset.csv').drop(columns='Unnamed: 0')
#print(alc_df)
county_grouped = alc_df.groupby(['County'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
#print(county_grouped.sort_values(by=['Bottles Sold'], ascending=False))

polk_2019_df = alc_df[(alc_df['Year'] == 2019) & (alc_df['County'] == 'POLK')]
polk_2020_df = alc_df[(alc_df['Year'] == 2020) & (alc_df['County'] == 'POLK')]
#polk_2018_df["Week"] = polk_2018_df["Date"].apply(get_week)
polk_2019_df["Week"] = polk_2019_df["Date"].apply(get_week)
polk_2020_df["Week"] = polk_2020_df["Date"].apply(get_week)
#print(polk_2020_df)
#print(polk_2020_df.dtypes)
#week_grouped_2018 = polk_2018_df.groupby(['Week'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
week_grouped_2019 = polk_2019_df.groupby(['Week'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()
week_grouped_2020 = polk_2020_df.groupby(['Week'], as_index=False)['Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)'].sum()

#print(week_grouped_2018)
print(week_grouped_2019)
print(week_grouped_2020)

print(week_grouped_2019["Bottles Sold"].mean())
print(week_grouped_2020["Bottles Sold"].mean())
#print(week_grouped)

# # 2018 plots
# x_axis = week_grouped_2018['Week']
# y_axis1 = week_grouped_2019['Bottles Sold']
# y_axis2 = week_grouped_2018['Volume Sold (Liters)']
# y_axis3 = week_grouped_2018['Sale (Dollars)']
# plt.plot(x_axis, y_axis1)
# plt.xlabel('Week')
# plt.ylabel('Bottles Sold')
# plt.title("Polk County 2018")
# plt.show()

# plt.plot(x_axis, y_axis2)
# plt.xlabel('Week')
# plt.ylabel('Volume Sold (Liters)')
# plt.title("Polk County 2018")
# plt.show()

# plt.plot(x_axis, y_axis3)
# plt.xlabel('Week')
# plt.ylabel('Sale (Dollars)')
# plt.title("Polk County 2018")
# plt.show()

# # 2019 plots
# x_axis = week_grouped_2019['Week']
# y_axis1 = week_grouped_2019['Bottles Sold']
# y_axis2 = week_grouped_2019['Volume Sold (Liters)']
# y_axis3 = week_grouped_2019['Sale (Dollars)']
# plt.plot(x_axis, y_axis1)
# plt.xlabel('Week')
# plt.ylabel('Bottles Sold')
# plt.title("Polk County 2019")
# plt.show()

# plt.plot(x_axis, y_axis2)
# plt.xlabel('Week')
# plt.ylabel('Volume Sold (Liters)')
# plt.title("Polk County 2019")
# plt.show()

# plt.plot(x_axis, y_axis3)
# plt.xlabel('Week')
# plt.ylabel('Sale (Dollars)')
# plt.title("Polk County 2019")
# plt.show()

# # 2020 plots
# x_axis = week_grouped_2020['Week']
# y_axis1 = week_grouped_2020['Bottles Sold']
# y_axis2 = week_grouped_2020['Volume Sold (Liters)']
# y_axis3 = week_grouped_2020['Sale (Dollars)']
# plt.plot(x_axis, y_axis1)
# plt.xlabel('Weeks (2020)')
# plt.ylabel('Bottles Sold')
# plt.title("Polk County 2020")
# plt.show()

# plt.plot(x_axis, y_axis2)
# plt.xlabel('Weeks (2020)')
# plt.ylabel('Volume Sold (Liters)')
# plt.title("Polk County 2020")
# plt.show()

# plt.plot(x_axis, y_axis3)
# plt.xlabel('Weeks (2020)')
# plt.ylabel('Sale (Dollars)')
# plt.title("Polk County 2020")
# plt.show()