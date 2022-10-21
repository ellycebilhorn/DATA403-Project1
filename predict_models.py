import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
from sklearn.metrics import r2_score

'''
Predictive Models and Cross-val
'''

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
for L in range(10, 20000, 10):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
        r_squared_v = r2_score(y_test, y_hat)
print("BOTTLES SOLD BEST LAMBDA: ", best_lambda_1)
print("BOTTLES SOLD TRAINING ERROR: ", min_training_error)
print("BOTTLES SOLD TESTING ERROR: ", min_err)
print("BOTTLES SOLD R SQUARED: ", r_squared_v)

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
for L in range(10, 20000, 10):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
        r_squared_v = r2_score(y_test, y_hat)
print("SALES VOLUME BEST LAMBDA: ", best_lambda_1)
print("SALES VOLUME TRAINING ERROR: ", min_training_error)
print("SALES VOLUME TESTING ERROR: ", min_err)
print("SALES VOLUME R SQUARED: ", r_squared_v)

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
for L in range(10, 20000, 10):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
        r_squared_v = r2_score(y_test, y_hat)

print("VOLUME SOLD BEST LAMBDA: ", best_lambda_1)
print("VOLUME SOLD TRAINING ERROR: ", min_training_error)
print("VOLUME SOLD TESTING ERROR: ", min_err)
print("VOLUME SOLD R SQUARED: ", r_squared_v)

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
for L in range(10, 20000, 10):
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    #y_hat = ols_regression(X_train, y_train)
    training_err = average_pct_err(y_train, y_hat)
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    #y_hat = ols_regression(X_test, y_test)
    testing_err = average_pct_err(y_test, y_hat)
    if testing_err < min_err:
        best_lambda_1 = L
        min_err = testing_err
        min_training_error = training_err
        r_squared_v = r2_score(y_test, y_hat)

print("STATE PROFIT BEST LAMBDA: ", best_lambda_1)
print("STATE PROFIT TRAINING ERROR: ", min_training_error)
print("STATE PROFIT TESTING ERROR: ", min_err)
print("STATE PROFIT R SQUARED: ", r_squared_v)

# ALL BUT ONE CROSS VALIDATION:
##################### Bottles Sold per capita #####################
Y = grouped_df['Bottles Sold per capita']

total_training_error = 0
total_testing_error = 0
MSE = 0
r2 = 0
for i in range(bottles_sold_df.shape[0]):
    y_train = Y.drop(i)
    y_test = Y[i]
    X_train = bottles_sold_df.drop(i)
    X_test = bottles_sold_df.iloc[i]
    L = 7010
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    err = average_pct_err(y_train, y_hat)
    total_training_error += err

    # Testing Set
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    err = average_pct_err(y_test, y_hat)
    total_testing_error += err
    curr_MSE = y_test - y_hat
    curr_MSE *= curr_MSE
    MSE += curr_MSE
print("AVERAGE BOTTLES SOLD TRAINING ERROR: ", total_training_error/bottles_sold_df.shape[0])
print("AVERAGE BOTTLES SOLD TESTING ERROR: ", total_testing_error/bottles_sold_df.shape[0])
print("AVERAGE BOTTLES SOLD MSE: ", MSE/bottles_sold_df.shape[0])

##################### Sales volume per capita #####################

Y = grouped_df['Sales per capita (Dollars)']
total_training_error = 0
total_testing_error = 0
MSE = 0
r2 = 0
for i in range(sales_volume_df.shape[0]):
    y_train = Y.drop(i)
    y_test = Y[i]
    X_train = sales_volume_df.drop(i)
    X_test = sales_volume_df.iloc[i]
    L = 8220
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    err = average_pct_err(y_train, y_hat)
    total_training_error += err

    # Testing Set
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    err = average_pct_err(y_test, y_hat)
    total_testing_error += err
    
    curr_MSE = y_test - y_hat
    MSE += curr_MSE

print("AVERAGE SALES VOLUME TRAINING ERROR: ", total_training_error/sales_volume_df.shape[0])
print("AVERAGE SALES VOLUME TESTING ERROR: ", total_testing_error/sales_volume_df.shape[0])
print("AVERAGE SALES VOLUME MSE: ", MSE/sales_volume_df.shape[0])

#####################  Volume of sold alcohol per capita #####################
Y = grouped_df['Liters Sold per capita']
total_training_error = 0
total_testing_error = 0
MSE = 0
r2 = 0
for i in range(volume_df.shape[0]):
    y_train = Y.drop(i)
    y_test = Y[i]
    X_train = volume_df.drop(i)
    X_test = volume_df.iloc[i]
    L = 8600
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    err = average_pct_err(y_train, y_hat)
    total_training_error += err

    # Testing Set
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    err = average_pct_err(y_test, y_hat)
    total_testing_error += err
    curr_MSE = y_test - y_hat
    MSE += curr_MSE

print("AVERAGE VOLUME SOLD TRAINING ERROR: ", total_training_error/volume_df.shape[0])
print("AVERAGE VOLUME SOLD TESTING ERROR: ", total_testing_error/volume_df.shape[0])
print("AVERAGE VOLUME SOLD MSE: ", MSE/volume_df.shape[0])

##################### State Profit per capita #####################
Y = grouped_df['Sales per capita (Dollars)']
total_training_error = 0
total_testing_error = 0
MSE = 0
r2 = 0
for i in range(state_profit_df.shape[0]):
    y_train = Y.drop(i)
    y_test = Y[i]
    X_train = state_profit_df.drop(i)
    X_test = state_profit_df.iloc[i]
    L = 12940
    betas1 = ridge_betas(X_train, y_train, L)
    y_hat = ridge_regression(X_train, y_train, L, betas1)
    err = average_pct_err(y_train, y_hat)
    total_training_error += err

    # Testing Set
    y_hat = ridge_regression(X_test, y_test, L, betas1)
    err = average_pct_err(y_test, y_hat)
    total_testing_error += err
    curr_MSE = y_test - y_hat
    MSE += curr_MSE
print("AVERAGE STATE PROFIT TRAINING ERROR: ", total_training_error/state_profit_df.shape[0])
print("AVERAGE STATE PROFIT TESTING ERROR: ", total_testing_error/state_profit_df.shape[0])
print("AVERAGE STATE PROFIT MSE: ", MSE/state_profit_df.shape[0])