# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 2018

@author: Toni Huovinen
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
dataset = pd.read_csv("Iris.csv")

# Seperate data into X and y
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4]

# Encoding y. Flower Species to dummy variables
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y) # This would be good in some situations

# Convert the values into dummy variables
encoded_y = np_utils.to_categorical(encoded_y)

# Splitting the data into training and test set.
X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Neural Network
classifier = Sequential()

# Input layer and hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=4))

# Output layer
classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))

# Compile the classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
classifier.fit(X_train, y_train, epochs=200, batch_size=5)

loss, accuracy = classifier.evaluate(X_test, y_test)
print(accuracy)






