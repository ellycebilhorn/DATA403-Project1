import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
    return np.mean((abs(Y-y_hat))/abs(Y))

alc_df = pd.read_csv('alc_dataset.csv', nrows=1000000)
#alc_df = pd.read_csv('alc_dataset.csv')
alc_df["Month"] = alc_df["Date"].apply(get_month)
alc_df = alc_df[alc_df.Year > 2017]
print(alc_df)

#alc_df.drop(columns = ["Year", "Date", "Unnamed: 0"], inplace = True)

# print(alc_df)
# print(alc_df.dtypes)

# #################### BOTTLES SOLD PER CAPITA ####################
# grouped_df_bottle = alc_df.drop(columns=['Vendor Name', "Category Name", "State Bottle Cost", "State Bottle Retail", "Sale (Dollars)"])
# grouped_df_bottle = pd.get_dummies(grouped_df_bottle, columns=['Store Name']).groupby(['Month', 'County', 'Population']).sum().reset_index()


# grouped_df_bottle.drop(columns=['County', 'Month'], inplace=True)

# grouped_df_bottle['Bottles Sold per capita'] = grouped_df_bottle['Bottles Sold'] / grouped_df_bottle['Population']
# print(grouped_df_bottle)

# # Bottles Sold per capita
# Y = grouped_df_bottle['Bottles Sold per capita']

# bottles_sold_df = grouped_df_bottle.drop(columns=['Bottles Sold per capita',  'Bottles Sold', "Bottle Volume (ml)", 'Volume Sold (Liters)'])
# normalize = ['Population']
# bottles_sold_df[normalize] = StandardScaler().fit_transform(bottles_sold_df[normalize])
# X_train, X_test, y_train, y_test = train_test_split(bottles_sold_df, Y, test_size=0.2, random_state = 42)


# # Training Set
# L = 10000
# betas1 = ridge_betas(X_train, y_train, L)

# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("BOTTLES SOLD TRAINING ERROR: ", err)

# # Testing Set
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)

# print("BOTTLES SOLD TESTING ERROR: ", err)

# #################### SALES VOLUME PER CAPITA ####################
# grouped_df_sales = alc_df.drop(columns=["State Bottle Cost", "State Bottle Retail", "Category Name", "Vendor Name"])
# grouped_df_sales = pd.get_dummies(grouped_df_sales, columns=['Store Name']).groupby(['Month', 'County', 'Population']).sum().reset_index()
# grouped_df_sales.drop(columns=['County', 'Month'], inplace=True)

# Y = grouped_df_sales['Sale (Dollars)'] / grouped_df_sales['Population']

# sales_df = grouped_df_sales.drop(columns=['Sale (Dollars)'])
# normalize = ['Population', "Bottle Volume (ml)", "Bottles Sold", "Volume Sold (Liters)"]
# sales_df[normalize] = StandardScaler().fit_transform(sales_df[normalize])
# X_train, X_test, y_train, y_test = train_test_split(sales_df, Y, test_size=0.2, random_state = 42)


# # Training Set
# L = 10000
# betas1 = ridge_betas(X_train, y_train, L)

# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("SALES VOLUME TRAINING ERROR: ", err)

# # Testing Set
# #L = 2000
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)

# print("SALES VOLUME TESTING ERROR: ", err)

# #################### STATE PROFIT PER CAPITA ####################
# grouped_profit_df = alc_df.drop(columns = ["Store Name"])
# grouped_profit_df = pd.get_dummies(grouped_profit_df, columns = ["Category Name", "Vendor Name"]).groupby(['Month', 'County', 'Population']).sum().reset_index().drop(columns=["Month", "County"])
# grouped_profit_df
# Y = (grouped_profit_df['State Bottle Retail'] - grouped_profit_df['State Bottle Cost'])/grouped_profit_df['Population']

# profit_df = grouped_profit_df.drop(columns=['State Bottle Retail', 'State Bottle Cost'])
# normalize = ['Population', "Bottles Sold", "Volume Sold (Liters)", "Sale (Dollars)"]
# profit_df[normalize] = StandardScaler().fit_transform(profit_df[normalize])
# X_train, X_test, y_train, y_test = train_test_split(profit_df, Y, test_size=0.2, random_state = 42)


# # Training Set
# L = 10000
# betas1 = ridge_betas(X_train, y_train, L)

# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("STATE PROFIT TRAINING ERROR: ", err)

# # Testing Set
# #L = 2000
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)

# print("STATE PROFIT TESTING ERROR: ", err)

# #################### VOLUME SOLD PER CAPITA ####################
# grouped_volume_df = alc_df.drop(columns = ["Bottle Volume (ml)", "Vendor Name", "Category Name", "State Bottle Cost", "State Bottle Retail"])
# grouped_volume_df = pd.get_dummies(grouped_volume_df, columns = ["Store Name"]).groupby(['Month', 'County', 'Population']).sum().reset_index().drop(columns=["Month", "County"])
# grouped_volume_df

# Y = grouped_volume_df['Volume Sold (Liters)'] / grouped_volume_df['Population']

# volume_df = grouped_volume_df.drop(columns=['Volume Sold (Liters)'])
# #print(volume_df)
# normalize = ['Population', "Bottles Sold", "Sale (Dollars)"]
# volume_df[normalize] = StandardScaler().fit_transform(volume_df[normalize])
# X_train, X_test, y_train, y_test = train_test_split(volume_df, Y, test_size=0.2, random_state = 42)


# # Training Set
# L = 10000
# betas1 = ridge_betas(X_train, y_train, L)

# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("VOLUME SOLD TRAINING ERROR: ", err)

# # Testing Set
# #L = 2000
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)

# print("VOLUME TESTING ERROR: ", err)