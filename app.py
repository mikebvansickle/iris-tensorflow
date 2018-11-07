import pandas as pd
import numpy as np
import requests
import re
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, normalize #1
from sklearn.model_selection import train_test_split #2
from datetime import datetime

def forward_propagation(x):
    #Hidden layer 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #Hidden layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#Download the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
r = requests.get(url, allow_redirects=True)
filename = "raw.csv"
open(filename, 'wb').write(r.content)

#Load data into memory
dataset = pd.read_csv('raw.csv', header=None, names=['sepal_length',
                                                     'sepal_width',
                                                     'petal_length',
                                                     'petal_width',
                                                     'species'])
dataset.head()

##Plot dataset to visualize
##NOTE: "size" parameter has been deprecated, now uses "height" instead
#seaborn.pairplot(dataset, hue="species", height=2, diag_kind="kde")
#plt.show()

#1
species_lb = LabelBinarizer()
Y = species_lb.fit_transform(dataset.species.values)

FEATURES = dataset.columns[0:4]
X_data = dataset[FEATURES].as_matrix()
X_data = normalize(X_data)

#2
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=1)
X_train.shape

#Model Parameters
learning_rate = 0.01
training_epochs = 100

#Neural Network Parameters
n_hidden_1 = 256 #1st layer number of neurons
n_hidden_2 = 128 #2nd layer number of neurons
n_input = X_train.shape[1] #input shape (105, 4)
n_classes = y_train.shape[1] #classes to predict

#Inputs
X = tf.placeholder("float",
                    shape=[None, n_input])
y = tf.placeholder("float",
                    shape=[None, n_classes])

#Dictionary of Weights and Biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#Model Outputs
yhat = forward_propagation(X)
ypredict = tf.argmax(yhat, axis=1)

#Backward propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train_op = optimizer.minimize(cost)

#Tensorflow sessions to train neural network
#Initializing variables
init = tf.global_variables_initializer()
#3
startTime = datetime.now()

with tf.Session() as sess:
    sess.run(init)

    #Epochs
    for epoch in range(training_epochs):
        for i in range(len(X_train)):
            summary = sess.run(train_op, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})

        train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X: X_train, y: y_train}))
        test_accuracy = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X: X_test, y: y_test}))
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()
print("Time elapsed:", datetime.now() - startTime)
