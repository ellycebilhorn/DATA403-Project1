import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
from sklearn.metrics import r2_score



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

def average_pct_err(Y, y_hat):
    return np.mean((abs(Y-y_hat))/abs(Y))

grouped_df = pd.read_csv('grouped_df.csv')
#grouped_df = grouped_df.drop("Unnamed: 0")

grouped_df['State Profit per capita'] = (grouped_df['State Bottle Retail'] - grouped_df['State Bottle Cost'])/grouped_df['Population']
grouped_df['Liters Sold per capita'] = grouped_df['Volume Sold (Liters)'] / grouped_df['Population']
grouped_df['Sales per capita (Dollars)'] = grouped_df['Sale (Dollars)'] / grouped_df['Population']
grouped_df['Bottles Sold per capita'] = grouped_df['Bottles Sold'] / grouped_df['Population']
grouped_df = grouped_df.drop(columns='Unnamed: 0')

##################### Bottles Sold per capita #####################
scaler = StandardScaler()

Y = grouped_df['Bottles Sold per capita']
# bottles_sold_df = grouped_df.drop(columns=['Bottles Sold per capita'])

bottles_sold_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Bottles Sold'])
bottles_sold_df = bottles_sold_df.drop(columns=['Bottle Volume (ml)', 'State Bottle Cost', 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)'])
bottles_sold_df = bottles_sold_df.drop(columns='Population')
print(bottles_sold_df)
#bottles_sold_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Bottles Sold'])
bottles_columns = pd.Series(bottles_sold_df.columns)
#bottles_sold_df = scaler.fit_transform(bottles_sold_df)
#normalize = ['Population', 'Bottle Volume (ml)', 'State Bottle Cost', 'Sale (Dollars)', 'Volume Sold (Liters)', "State Bottle Retail"]
#normalize = ['Population']
#bottles_sold_df[normalize] = StandardScaler().fit_transform(bottles_sold_df[normalize])
#print(bottles_sold_df)
X_train, X_test, y_train, y_test = train_test_split(bottles_sold_df, Y, test_size=0.2, random_state = 42)
# X_train = np.matrix(X_train)
# print("X TRIANING: ", X_train)
# X_test = np.matrix(X_test)
# y_train = np.matrix(y_train)
# y_test = np.matrix(y_test)
# Training Set
best_lambda_1 = 0
min_training_error = 0
min_err = 1000000
#for L in range(10, 20000, 10):
for L in range(7010, 7011):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    r_squared_v = r2_score(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
print("BOTTLES SOLD BEST LAMBDA: ", best_lambda_1)
print("BOTTLES SOLD TRAINING ERROR: ", min_training_error)
print("BOTTLES SOLD TESTING ERROR: ", min_err)
print("BOTTLES SOLD R SQUARED: ", r_squared_v)

# L = 19990
# betas1 = ridge_betas(X_train, y_train, L)
# print("BOTTLES SOLD VARIANCE: ", np.var(betas1))
# bottles_coefs = pd.concat([bottles_columns, pd.Series(abs(betas1))], axis=1).sort_values(by=1, ascending=False)
# y_hat = ridge_regression(X_train, y_train, L, betas1)
# #y_hat = ols_regression(X_train, y_train)
# err = average_pct_err(y_train, y_hat)
# print("BOTTLES SOLD TRAINING ERROR: ", err)

# # Testing Set
# #L = 2000
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# #y_hat = ols_regression(X_test, y_test)
# err = average_pct_err(y_test, y_hat)
# #print(y_test, y_hat)
# #print(y_test, y_hat)
# print("BOTTLES SOLD TESTING ERROR: ", err)


##################### Sales volume per capita #####################
scaler = StandardScaler()

Y = grouped_df['Sales per capita (Dollars)']
# bottles_sold_df = grouped_df.drop(columns=['Bottles Sold per capita'])

sales_volume_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Sale (Dollars)'])
sales_volume_df = sales_volume_df.drop(columns=['Bottle Volume (ml)', 'State Bottle Cost', 'State Bottle Retail', 'Bottles Sold', 'Volume Sold (Liters)', 'Population'])
sales_volume_columns = pd.Series(sales_volume_df.columns)
#normalize = ['Population']
#sales_volume_df[normalize] = StandardScaler().fit_transform(sales_volume_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(sales_volume_df, Y, test_size=0.2, random_state = 42)

# Training Set
best_lambda_1 = 0
min_training_error = 0
min_err = 1000000
#for L in range(10, 20000, 10):
for L in range(8220, 8221):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    r_squared_v = r2_score(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
print("SALES VOLUME BEST LAMBDA: ", best_lambda_1)
print("SALES VOLUME TRAINING ERROR: ", min_training_error)
print("SALES VOLUME TESTING ERROR: ", min_err)
print("SALES VOLUME R SQUARED: ", r_squared_v)
# L = 1000
# betas1 = ridge_betas(X_train, y_train, L)
# print("SALES VOLUME VARIANCE: ", np.var(betas1))

# #sales_coef = pd.concat([sales_volume_columns, pd.Series(betas1)], axis=1)
# #sales_coef = new_df.sort_values(by=1, ascending=False)
# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("SALES VOLUME TRAINING ERROR: ", err)

# # Testing Set
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)
# #print(y_test, y_hat)
# print("SALES VOLUME TESTING ERROR: ", err)

#####################  Volume of sold alcohol per capita #####################
scaler = StandardScaler()

Y = grouped_df['Liters Sold per capita']

volume_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'Volume Sold (Liters)', 'Bottle Volume (ml)'])
volume_df = volume_df.drop(columns=['Sale (Dollars)', 'State Bottle Cost', 'State Bottle Retail', 'Bottles Sold', 'Population'])

#print(volume_df)
volume_columns = pd.Series(volume_df.columns)
#normalize = ['Population']
#volume_df[normalize] = StandardScaler().fit_transform(volume_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(volume_df, Y, test_size=0.2, random_state = 42)

# Training Set
best_lambda_1 = 0
min_training_error = 0
min_err = 1000000
#for L in range(10, 20000, 10):
for L in range(8600, 8601):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    r_squared_v = r2_score(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
print("VOLUME SOLD BEST LAMBDA: ", best_lambda_1)
print("VOLUME SOLD TRAINING ERROR: ", min_training_error)
print("VOLUME SOLD TESTING ERROR: ", min_err)
print("VOLUME SOLD R SQUARED: ", r_squared_v)
# L = 10000
# betas1 = ridge_betas(X_train, y_train, L)
# print("VOLUME SOLD VARIANCE: ", np.var(betas1))

# new_df = pd.concat([volume_columns, pd.Series(betas1)], axis=1)
# #print(new_df.sort_values(by=1, ascending=False))
# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("VOLUME SOLD TRAINING ERROR: ", err)

# # Testing Set
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)
# #print(y_test, y_hat)
# print("VOLUME SOLD TESTING ERROR: ", err)


##################### State Profit per capita #####################
scaler = StandardScaler()

Y = grouped_df['Sales per capita (Dollars)']
state_profit_df = grouped_df.drop(columns=['Bottles Sold per capita', 'Sales per capita (Dollars)', 'Liters Sold per capita', 'State Profit per capita', 'State Bottle Retail', 'State Bottle Cost'])
state_profit_df = state_profit_df.drop(columns=['Sale (Dollars)', 'Volume Sold (Liters)', 'Bottle Volume (ml)', 'Bottles Sold', 'Population'])

#print(state_profit_df)
profit_columns = pd.Series(state_profit_df.columns)
#normalize = ['Population']
#state_profit_df[normalize] = StandardScaler().fit_transform(state_profit_df[normalize])
X_train, X_test, y_train, y_test = train_test_split(state_profit_df, Y, test_size=0.2, random_state = 10)

# Training Set
best_lambda_1 = 0
min_training_error = 0
min_err = 1000000
#for L in range(10, 20000, 10):
for L in range(12940, 12941):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    r_squared_v = r2_score(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
print("STATE PROFIT BEST LAMBDA: ", best_lambda_1)
print("STATE PROFIT TRAINING ERROR: ", min_training_error)
print("STATE PROFIT TESTING ERROR: ", min_err)
print("STATE PROFIT R SQUARED: ", r_squared_v)
# L = 10000
# betas1 = ridge_betas(X_train, y_train, L)
# print("STATE PROFIT VARIANCE: ", np.var(betas1))

# new_df = pd.concat([profit_columns, pd.Series(betas1)], axis=1)
# #print(new_df.sort_values(by=1, ascending=False))
# y_hat = ridge_regression(X_train, y_train, L, betas1)
# err = average_pct_err(y_train, y_hat)
# print("STATE PROFIT TRAINING ERROR: ", err)

# # Testing Set
# L = 100
# y_hat = ridge_regression(X_test, y_test, L, betas1)
# err = average_pct_err(y_test, y_hat)
# #print(y_test, y_hat)
# print("STATE PROFIT TESTING ERROR: ", err)