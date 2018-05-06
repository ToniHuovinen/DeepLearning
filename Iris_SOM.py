# -*- coding: utf-8 -*-
"""
Created on Wed Apr 4 2018

@author: Toni Huovinen
"""

from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt

# Get the data from csv
data = np.genfromtxt('Iris.csv', delimiter = ',', usecols = (0,1,2,3))

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(data)

# Initialization and Training
som = MiniSom(x = 7, y = 7, input_len = 4, sigma = 1.0, learning_rate=0.5)
som.random_weights_init(X)
print("Training...")
som.train_random(data = X, num_iteration = 100)
print("Training ready!")

# Visualize
plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()

# Create targets and fit the markers on them
target = np.genfromtxt('Iris.csv', delimiter = ',', usecols = (4), dtype = str)
k = np.zeros(len(target), dtype = int)
k[target == 'Iris-setosa'] = 0
k[target == 'Iris-versicolor'] = 1
k[target == 'Iris-virginica'] = 2

# Create markers
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

for i, x in enumerate(X):
    w = som.winner(x) # Find the winner
    plt.plot(w[0] + 0.5,
             w[1] + 0.5,
             # Place markers on winner
             markers[k[i]],
             markeredgecolor = colors[k[i]],
             markerfacecolor = 'None',
             markersize = 10,
             markeredgewidth = 2)

plt.axis([0, 7, 0, 7])
plt.show()

# List the flowers we are interested in. You can check the coordinates from
# map and place them on the mappings below
mappings = som.win_map(X)
flowers = np.concatenate((mappings[(2,4)], mappings[(0,6)]), axis = 0)
flowers = sc.inverse_transform(flowers)






