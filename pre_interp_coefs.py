import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
File that generates coefficient weightings for the interpretive models.
'''

def get_month(string):
    return string.split("/")[0]

def coef_mat(X, Y):
    xtx = np.linalg.inv(np.matmul(X.T, X))
    print(np.matmul(X.T, X).shape)
    print(np.matmul(X.T, X))
    beta_hat = xtx * (X.transpose() * Y.transpose())
    print(beta_hat)
    return beta_hat

def ridge_betas(X, Y, L):
    xtx = np.linalg.inv(X.T.dot(X) + np.identity(X.shape[1])*L)
    beta_hat = xtx.dot(X.T.dot(Y))
    return beta_hat

def ols_regression(X, Y):
    beta_hat = coef_mat(X, Y)
    return X * beta_hat

def ridge_regression(X, Y, L, beta_hat):
    return X.dot(beta_hat)

def lse(Y, y_hat):
    return (Y - y_hat).T.dot(Y - y_hat)

def rss(Y, y_hat):
    return str(np.sum(
        np.square(Y - y_hat)
    ))

def average_pct_err(Y, y_hat):
    return np.mean((abs(Y-y_hat))/Y)

alc_df = pd.read_csv('alc_dataset.csv', nrows=3000000)
alc_df["Month"] = alc_df["Date"].apply(get_month)
alc_df.drop(columns = ["Year", "Date", "Unnamed: 0"], inplace = True)

grouped_df = pd.get_dummies(alc_df, columns=['Store Name', 'Category Name', 'Vendor Name']).groupby(['Month', 'County', 'Population']).sum().reset_index()
grouped_df = pd.get_dummies(grouped_df, columns=['Month', 'County'])
grouped_df['State Profit per capita'] = (grouped_df['State Bottle Retail'] - grouped_df['State Bottle Cost'])/grouped_df['Population']
grouped_df['Liters Sold per capita'] = grouped_df['Volume Sold (Liters)'] / grouped_df['Population']
grouped_df['Sales per capita (Dollars)'] = grouped_df['Sale (Dollars)'] / grouped_df['Population']
grouped_df['Bottles Sold per capita'] = grouped_df['Bottles Sold'] / grouped_df['Population']

# Bottles Sold per capita
scaler = StandardScaler()

Y = grouped_df['Bottles Sold per capita']
bottles_sold_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Bottles Sold'])
bottles_columns = pd.Series(bottles_sold_df.columns)
normalize = ['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'Sale (Dollars)', 'Volume Sold (Liters)', "State Bottle Retail"]
bottles_sold_df[normalize] = StandardScaler().fit_transform(bottles_sold_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(bottles_sold_df, Y, test_size=0.2, random_state = 42)

# Training Set
L = 10000
betas1 = ridge_betas(X_train, y_train, L)

bottles_coefs = pd.concat([bottles_columns, pd.Series(abs(betas1))], axis=1).sort_values(by=1, ascending=False)
y_hat = ridge_regression(X_train, y_train, L, betas1)
err = average_pct_err(y_train, y_hat)
print("TRAINING ERROR: ", err)

# Testing Set
y_hat = ridge_regression(X_test, y_test, L, betas1)
err = average_pct_err(y_test, y_hat)
print("TESTING ERROR: ", err)

# Sales volume per capita
scaler = StandardScaler()

Y = grouped_df['Sales per capita (Dollars)']
# bottles_sold_df = grouped_df.drop(columns=['Bottles Sold per capita'])

sales_volume_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Sale (Dollars)'])
sales_volume_columns = pd.Series(sales_volume_df.columns)
normalize = ['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'Bottles Sold', 'Volume Sold (Liters)']
sales_volume_df[normalize] = StandardScaler().fit_transform(sales_volume_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(sales_volume_df, Y, test_size=0.2, random_state = 42)

# Training Set
L = 1000
betas1 = ridge_betas(X_train, y_train, L)

sales_coef = pd.concat([sales_volume_columns, pd.Series(betas1)], axis=1)
sales_coef = sales_coef.sort_values(by=1, ascending=False)
y_hat = ridge_regression(X_train, y_train, L, betas1)
err = average_pct_err(y_train, y_hat)
print("TRAINING ERROR: ", err)

# Testing Set
y_hat = ridge_regression(X_test, y_test, L, betas1)
err = average_pct_err(y_test, y_hat)
#print(y_test, y_hat)
print("TESTING ERROR: ", err)

# Volume of sold alcohol per capita
scaler = StandardScaler()

Y = grouped_df['Liters Sold per capita']
volume_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Volume Sold (Liters)', 'Bottle Volume (ml)'])
volume_columns = pd.Series(volume_df.columns)
normalize = ['Population', 'State Bottle Cost', 'Bottles Sold', 'State Bottle Retail', 'Sale (Dollars)']
volume_df[normalize] = StandardScaler().fit_transform(volume_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(volume_df, Y, test_size=0.2, random_state = 42)
# Training Set
L = 10000
betas1 = ridge_betas(X_train, y_train, L)

volume_coef = pd.concat([volume_columns, pd.Series(betas1)], axis=1)
y_hat = ridge_regression(X_train, y_train, L, betas1)
err = average_pct_err(y_train, y_hat)
print("TRAINING ERROR: ", err)
# Testing Set
y_hat = ridge_regression(X_test, y_test, L, betas1)
err = average_pct_err(y_test, y_hat)
print("TESTING ERROR: ", err)

# State Profit per capita
scaler = StandardScaler()

Y = grouped_df['Sales per capita (Dollars)']
state_profit_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'State Bottle Retail', 'State Bottle Cost'])
profit_columns = pd.Series(state_profit_df.columns)
normalize = ['Population', 'Bottle Volume (ml)', 'Bottles Sold', 'Volume Sold (Liters)', 'Sale (Dollars)']
state_profit_df[normalize] = StandardScaler().fit_transform(state_profit_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(state_profit_df, Y, test_size=0.2, random_state = 10)

# Training Set
L = 10000
betas1 = ridge_betas(X_train, y_train, L)

profit_coef = pd.concat([profit_columns, pd.Series(betas1)], axis=1)
y_hat = ridge_regression(X_train, y_train, L, betas1)
err = average_pct_err(y_train, y_hat)
print("TRAINING ERROR: ", err)

# Testing Set
L = 10000
y_hat = ridge_regression(X_test, y_test, L, betas1)
err = average_pct_err(y_test, y_hat)
#print(y_test, y_hat)
print("TESTING ERROR: ", err)

############### MODELING THE COEFFICIENTS
def get_types(string):
    the_type = string.split("_")[0]
    if the_type == "Category Name":
        return "Category"
    elif the_type == "Vendor Name":
        return "Vendor"
    elif the_type == "Store Name":
        return "Store"
    elif the_type == "County":
        return "County"
    elif the_type == "Month":
        return "Month"
    else:
        return "Numerical"

bottles_coefs = bottles_coefs.rename(columns = {0: "Coefficient", 1: "Weight"})
bottles_coefs["Weight"] = bottles_coefs["Weight"].apply(abs)
bottles_coefs["Type"] = bottles_coefs.iloc[:, 0].apply(get_types)
bottle_cat_num = bottles_coefs[bottles_coefs["Type"] == "Numerical"]
print(bottles_coefs.groupby("Type").mean())
print(bottles_coefs[bottles_coefs["Type"] == "Store"].sort_values(by="Weight", ascending = False).iloc[:5])
print(bottles_coefs[bottles_coefs["Type"] == "Category"].sort_values(by="Weight", ascending = False).iloc[:5])

sales_coef = sales_coef.rename(columns = {0: "Coefficient", 1: "Weight"})
sales_coef["Weight"] = sales_coef["Weight"].apply(abs)
sales_coef["Type"] = sales_coef.iloc[:, 0].apply(get_types)
sales_cat_num = sales_coef[sales_coef["Type"] == "Numerical"]
print(sales_coef.groupby("Type").mean())
print(sales_coef[sales_coef["Type"] == "Store"].sort_values(by="Weight", ascending = False).iloc[:5])
print(sales_coef[sales_coef["Type"] == "Category"].sort_values(by="Weight", ascending = False).iloc[:5])
print(sales_coef[sales_coef["Type"] == "Month"].sort_values(by="Weight", ascending = False).iloc[:10])

volume_coef = volume_coef.rename(columns = {0: "Coefficient", 1: "Weight"})
volume_coef["Weight"] = volume_coef["Weight"].apply(abs)
volume_coef["Type"] = volume_coef.iloc[:, 0].apply(get_types)
volume_cat_num = volume_coef[volume_coef["Type"] == "Numerical"]
print(volume_coef.groupby("Type").mean())
print(volume_coef[volume_coef["Type"] == "Store"].sort_values(by="Weight", ascending = False).iloc[:5])
print(volume_coef[volume_coef["Type"] == "Category"].sort_values(by="Weight", ascending = False).iloc[:6])

profit_coef = profit_coef.rename(columns = {0: "Coefficient", 1: "Weight"})
profit_coef["Weight"] = profit_coef["Weight"].apply(abs)
profit_coef["Type"] = profit_coef.iloc[:, 0].apply(get_types)
profit_cat_num = profit_coef[profit_coef["Type"] == "Numerical"]
print(profit_coef.groupby("Type").mean())
print(profit_coef[profit_coef["Type"] == "Store"].sort_values(by="Weight", ascending = False).iloc[:5])
print(profit_coef[profit_coef["Type"] == "Category"].sort_values(by="Weight", ascending = False).iloc[:6])
