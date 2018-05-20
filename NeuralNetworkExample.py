# -*- coding: utf-8 -*-
"""
Created on Sat May  5 2018

@author: Toni Huovinen
"""

# Simple Regression Neural Network
# Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Variables, Placeholders, initialize features and neurodes
n_features = 10
n_dense_neurons = 3

# None means that dataset can grow down, but has n_features
x = tf.placeholder(tf.float32, (None, n_features))

# Weights
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))

# Bias
b = tf.Variable(tf.ones([n_dense_neurons]))

# Operations, matrix multiplication, addition and sigmoid as activation function
xW = tf.matmul(x, W)
z = tf.add(xW, b)
a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,n_features])})


# Create linear data and add some random variation to it 
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)

# Create label for the data so it becomes X vs Y. 
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)

# Plot the data as stars
plt.plot(x_data, y_label, '*')
plt.show()

# Solve y = m * x + b with neural network
# m as 0.32 was chosen in random
# b as 0.67 was chones in random. These numbers can be anything but it's best to choose between 0 - 1
m = tf.Variable(0.32)
b = tf.Variable(0.67)

# Cost function
# Initialize
error = 0

for x,y in zip(x_data, y_label):
    
    y_hat = m*x + b
    
    # Assign
    error += (y-y_hat)**2

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Create the Tensorflow graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    training_steps = 100
    
    for i in range(training_steps):
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m,b])


x_test = np.linspace(-1,11,10)

# y = mx + b
y_pred_plot = final_slope * x_test + final_intercept

# Create plots for both the line and and data
plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()



