import pandas as pd
import stats as stat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# open data file
data = pd.read_csv('machine.data')

data.columns = ['VendorName', 'ModelName', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

features = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']

print("Below are the correlations between each feature and the ERP attribute")

# set the corralations into a list
corr_list = {abs(stat.correlation(data[feature], data['ERP'])): feature for feature in features }

for key, value in corr_list.items():
    print("The correlation for ", value, " is ", key)


sortedCorrelation = [item for item, __ in corr_list.items()]

# sort the correlations
sortedCorrelation = sorted(sortedCorrelation, reverse=True)

# get the best correlations
top4 = [corr_list[sortedCorrelation[i]] for i in range(0, 4)]
top4.append('ERP')


top4Iters = [data.columns.get_loc(i) for i in top4]

print(top4Iters)
 
# cut up the dataset into data and targets
X = data.iloc[:,top4Iters[:-1]]
y = data.iloc[:, top4Iters[-1:]]

# get the training and testing sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# do the regression
model = LinearRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)

# get the needed data
mae = mean_absolute_error(y_test, predict)
mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mean_squared_error(y_test, predict))

print("The performance for testing set")
print("-------------------------------")
print('MAE is ', mae)
print('MSE is ', mse)
print('RMSE is ',rmse)