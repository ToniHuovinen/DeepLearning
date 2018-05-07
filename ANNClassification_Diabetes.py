# -*- coding: utf-8 -*-
"""
Created on Sun May 6 2018

@author: Toni Huovinen
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Loading the dataset with pandas
dataset = pd.read_csv("pima-indians-diabetes.csv")

# Using iloc to choose columns for the model
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Dataset has no categorical data that has any use for the model so we don't need any categorical encoding
# Splitting the dataset into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Feature Scaling
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)

# Fitting the Logistic Regression and creating the Classifier
# Initialize ANN
classifier = Sequential()

# Add input layer and first hidden layer. Activation Function Relu and on output Sigmoid
classifier.add(Dense(units=12, kernel_initializer= 'uniform', activation= 'relu', input_dim=8))
classifier.add(Dropout(rate = 0.1))

# Add second hidden layer. Activation Function Relu 
classifier.add(Dense(units=8, kernel_initializer= 'uniform', activation= 'relu'))
classifier.add(Dropout(rate = 0.1))

# Output Layer
classifier.add(Dense(units=1, kernel_initializer= 'uniform', activation= 'sigmoid'))

# Compiling the ANN,
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN into the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=200)

# Predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




















