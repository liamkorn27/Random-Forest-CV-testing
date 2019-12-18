  #------- 2019. 10.10 --------- #
import pandas as pd 
import numpy as np 

# read data from excel file
dataset = pd.read_excel(r'Original500samples.xlsx')
#remove molecule_id from data
dataset = dataset.drop('molecule_id', axis = 1)
#divide into dependent and independent variables
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:,0].values
print(Y)
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

model = RandomForestRegressor(n_estimators=50)

kf = KFold(n_splits=10)

# data pre-processing
col = len(X[0])
row = len(X)

print(X)
Z = [[0] * col for i in range(row)]   # initialize new 2d - array filled with 0.

for i in range(1,row - 1):
	for j in range(1,col):
		if(X[i][j] != X[i+1][j]):
			X[i][j] = abs(X[i][j] - X[i+1][j]) 
		else:
			X[i][j] = X[i][j]
index = 1

print(X)
for train_index, test_index in kf.split(X):

	print('\n','******************************',index,' -  Fold','******************************','\n')
	print('TRAIN : ', train_index, '\n\n', 'TEST : ', test_index, '\n')

	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]
	
	 # For training, fit() is used
	model.fit(X_train, Y_train)
 	
 	 # Default metric is R2 for regression, which can be accessed by score()
	scores = model.score(X_test, Y_test)

	 # For other metrics, we need the predictions of the model
	Y_pred = model.predict(X_test)

	 # evaluation part	
	print('Mean Absolute Error : ', metrics.mean_absolute_error(Y_test, Y_pred))
	print('Mean Squared Error : ', metrics.mean_squared_error(Y_test, Y_pred))
	print('Root Mean Squared Error : ', metrics.r2_score(Y_test, Y_pred), '\n')
	index = index + 1

