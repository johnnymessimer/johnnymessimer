import pandas as pd
from pandas_datareader import data as wb


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import keras
from keras.layers import Dense, Activation
from keras.models import Sequential



###################################################################################################

ticker_symbol = 'PEIX'
stock_price_data = wb.DataReader(ticker_symbol, data_source = 'yahoo', start = '2017-01-01')

 
data_curr = stock_price_data.iloc[1:-1]
data_prev = stock_price_data.iloc[0:-2]

 
data_curr.reset_index(level=0, inplace=True)
data_prev.reset_index(level=0, inplace=True)

 
data_curr['Year'] = data_curr['Date'].dt.year
data_curr['Month'] = data_curr['Date'].dt.month
data_curr['Day'] = data_curr['Date'].dt.day
 

data_curr = data_curr.drop(['Date','Adj Close'], axis=1)
data_prev = data_prev.drop(['Date','Adj Close'], axis=1)

 
data_curr.loc[data_curr.Year == 2017, 'Year'] = "0"
data_curr.loc[data_curr.Year == 2018, 'Year'] = "1"
data_curr.loc[data_curr.Year == 2019, 'Year'] = "2"

 
data_curr.loc[data_curr.Month == 1, 'Month'] = "1"
data_curr.loc[data_curr.Month == 2, 'Month'] = "1"
data_curr.loc[data_curr.Month == 3, 'Month'] = "1"
data_curr.loc[data_curr.Month == 4, 'Month'] = "2"
data_curr.loc[data_curr.Month == 5, 'Month'] = "2"
data_curr.loc[data_curr.Month == 6, 'Month'] = "2"
data_curr.loc[data_curr.Month == 7, 'Month'] = "3"
data_curr.loc[data_curr.Month == 8, 'Month'] = "3"
data_curr.loc[data_curr.Month == 9, 'Month'] = "3"
data_curr.loc[data_curr.Month == 10, 'Month'] = "4"
data_curr.loc[data_curr.Month == 11, 'Month'] = "4"
data_curr.loc[data_curr.Month == 12, 'Month'] = "4"


#data_curr.loc[data_curr.Day == 1, 'Day'] = "1"
 

data_curr['Volume'] = data_curr['Volume'].div(100000).round(2)
data_prev['Volume'] = data_prev['Volume'].div(100000).round(2)


data_curr.columns = ['High','Low','Open','Close','Volume', 'Year', 'Month', 'Day']
data_prev.columns = ['Prev_High','Prev_Low','Prev_Open','Prev_Close','Prev_Volume']


frames = [data_curr, data_prev]
matrix1 = pd.concat(frames, axis=1)
matrix1 = matrix1[['Prev_High','Prev_Low','Prev_Open','Prev_Close','Prev_Volume','High','Low','Open','Volume', 'Year', 'Month', 'Day', 'Close']]

 
X = matrix1.iloc[:,0:12]
X = X.astype('float64', copy=False)
X = X.values

 
y = matrix1.iloc[:,12:13]
y = y.astype('float64', copy=False)
y = y.values


#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#y = scaler.fit_transform(y)
 
######################################################################################################################
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg1.predict(X_test)

 
#degree = 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
poly_reg2 = LinearRegression()
poly_reg2.fit(x_poly, y)

 
#Support Vector Regression SVR
from sklearn.svm import SVR
svr_reg3 = SVR(kernel = 'rbf')
svr_reg3.fit(X, y)

 
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree_reg4 = DecisionTreeRegressor(random_state = 0)
tree_reg4.fit(X, y)

 
#Random Forest
from sklearn.ensemble import RandomForestRegressor
fore_reg5 = RandomForestRegressor(n_estimators = 100, random_state = 0)
fore_reg5.fit(X, y)
 

#########################################################################################################################

#ANN1 - Artificial Neural Network


# Initialising the ANN
ANN1 = Sequential()
# Adding the input layer and the first hidden layer
ANN1.add(Dense(32, activation = 'relu', input_dim=12))
# Adding the second hidden layer
ANN1.add(Dense(units = 32, activation = 'relu'))
# Adding the third hidden layer
ANN1.add(Dense(units = 32, activation = 'relu'))
# Adding the output layer
ANN1.add(Dense(units = 1))
 

# Compiling the ANN
#An optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the Optimizer class
#A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function
#For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.
ANN1.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

 
# Fitting the ANN to the Training set
ANN1.fit(X_train, y_train, batch_size = 10, epochs = 100)
 

#########################################################################################################################

#ANN2 with keras

 
#classifier = Sequential()
ANN2 = Sequential()
# Adding the input layer and the first hidden layer, output_dim = # of nodes, input_dim = # independent variables
#rectifier function = relu
ANN2.add(Dense(output_dim=6, init='uniform', activation = 'relu', input_dim=12))
# Second hidden layer
ANN2.add(Dense(output_dim=6, init='uniform', activation = 'relu'))
# Add the output layer using sigmoid or softmax
ANN2.add(Dense(output_dim=1, init='uniform', activation = 'sigmoid'))


# Compile the ANN
#An optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the Optimizer class
#A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function
#A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function.
ANN2.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])
 

# Fit the ANN to the Training set
#batch size = # of observations after which it will update weights
#epochs = # of times to retrain the data
ANN2.fit(X_train, y_train, batch_size = 100, epochs = 100)

 
#########################################################################################################################

today_numbers = stock_price_data.tail(2)
yesterday = today_numbers.iloc[0:1]
today = today_numbers.iloc[1:2]
today.reset_index(level=0, inplace=True)
yesterday.reset_index(level=0, inplace=True)

 
today['Year'] = today['Date'].dt.year
today['Month'] = today['Date'].dt.month
today['Day'] = today['Date'].dt.day

 
today = today.drop(['Date','Adj Close'], axis=1)
yesterday = yesterday.drop(['Date','Adj Close'], axis=1)

 
today.loc[today.Year == 2017, 'Year'] = "1"
today.loc[today.Year == 2018, 'Year'] = "2"
today.loc[today.Year == 2019, 'Year'] = "3"

 
today.loc[today.Month == 2, 'Month'] = "1"
today.loc[today.Month == 3, 'Month'] = "1"
today.loc[today.Month == 4, 'Month'] = "2"
today.loc[today.Month == 5, 'Month'] = "2"
today.loc[today.Month == 6, 'Month'] = "2"
today.loc[today.Month == 7, 'Month'] = "3"
today.loc[today.Month == 8, 'Month'] = "3"
today.loc[today.Month == 9, 'Month'] = "3"
today.loc[today.Month == 10, 'Month'] = "4"
today.loc[today.Month == 11, 'Month'] = "4"
today.loc[today.Month == 12, 'Month'] = "4"

 
today['Volume'] = today['Volume'].div(100000).round(2)
yesterday['Volume'] = yesterday['Volume'].div(100000).round(2)

 
today.columns = ['High','Low','Open','Close','Volume', 'Year', 'Month', 'Day']
yesterday.columns = ['Prev_High','Prev_Low','Prev_Open','Prev_Close','Prev_Volume']

 
frames = [today, yesterday]
matrix2 = pd.concat(frames, axis=1)
matrix2 = matrix2[['Prev_High','Prev_Low','Prev_Open','Prev_Close','Prev_Volume','High','Low','Open','Volume', 'Year', 'Month', 'Day', 'Close']]

 
today_numbers = matrix2.iloc[:,0:12]
today_numbers = today_numbers.astype('float64', copy=False)
today_numbers = today_numbers.values
#today_numbers = scaler.fit_transform(today_numbers)
 
today_close_pred1 = lin_reg1.predict(today_numbers)
today_close_pred2 = poly_reg2.predict(poly_reg.fit_transform(today_numbers))
today_close_pred3 = svr_reg3.predict(today_numbers)
today_close_pred4 = tree_reg4.predict(today_numbers)
today_close_pred5 = fore_reg5.predict(today_numbers)
today_close_pred6 = ANN1.predict(today_numbers)
today_close_pred7 = ANN2.predict(today_numbers)

 

