#Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing dataset

dataset=pd.read_csv('Data.csv')


#for object error
#"""dataset= dataset.astype(float)"""
#snp.set_printoptions(threshold = np.nan)


#Creating Independent variable matrices

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


#Taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])

X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)


#Splitting the data into training set and test set

#cross_validation  is changed into model selection
from sklearn.model_selection import train_test_split
X_train ,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


#Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)