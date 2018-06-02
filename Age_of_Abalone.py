# -*- coding: utf-8 -*-
"""
Created on Thu May 31 2018

@author: Toni Huovinen
"""

# Predicting the age of Abalone's
# Labels:
"""
Sex / nominal / -- / M, F, and I (infant)
Length / continuous / mm / Longest shell measurement
Diameter / continuous / mm / perpendicular to length
Height / continuous / mm / with meat in shell
Whole weight / continuous / grams / whole abalone
Shucked weight / continuous / grams / weight of meat
Viscera weight / continuous / grams / gut weight (after bleeding)
Shell weight / continuous / grams / after being dried
Rings / integer / -- / +1.5 gives the age in years
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense

# DATAPREPARATION
# Dataset
data = pd.read_csv("Abalone.csv")

# Seperate data to X and y
X = data.iloc[:,0:8].values
y = data.iloc[:,8].values

# Encoding the categorical values. Label first, then encode with OneHotEncoder
# 001 = male, 100 = female, 010 = infant
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the data into training and test set. Abalone Gender has been dropped for now.
# Not sure if gender brings any real value for predicting age.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Scaling, might not be needed. Values already are pretty near between 0 - 1
# Incase scaling is needed, just uncomment this section
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# MODEL
age_model = Sequential()

# Input layer and first hidden layer
age_model.add(Dense(units=12, kernel_initializer='uniform', activation='relu', input_dim=10))

# Second layer
age_model.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))

# Output. Since this is a regression model, we do not need activation function
age_model.add(Dense(units=1, kernel_initializer= 'uniform'))

# Compile and fit
age_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
age_model.fit(X_train, y_train, verbose=1, batch_size=10, epochs=50)

# Make predictions
y_pred = age_model.predict(X_test)

# Evaluate
score = age_model.evaluate(X_test, y_test, verbose=0)

print("\nAccuracy: ", score[1])









