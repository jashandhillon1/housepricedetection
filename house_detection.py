#importing the libraries
import pandas as pd
import numpy as np
import matplotlib as plt


#importing the dataset
dataset=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')
z=dataset2[:]
x=dataset.iloc[:, :-1]
y=dataset.iloc[:,-1]


#taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X_train)
X_train=imputer.transform(X_train)

#taking care of missing data of train set
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X_test)
X_test=imputer.transform(X_test)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0)


X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)
cols=[39,44,46,48,52,56,59,64,67]
x.drop(x.columns[cols],axis=1,inplace=True)

#taking care of categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x= LabelEncoder()
x.values[:,5] = labelencoder_x.fit_transform(x.values[:,5].astype('str'))
onehotencoder = OneHotEncoder(categorical_features = [5])
x = onehotencoder.fit_transform(x).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)
dataset2 = sc_X.transform(dataset2)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


#fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the results
pred=regressor.predict(X_test)

