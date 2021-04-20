
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

onehotencoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()

X_train = scaler.fit_transform(train.values[:,1:])
y_train = onehotencoder.fit_transform(train.values[:,:1]).toarray()

X_test = scaler.fit_transform(test.values[:,:])

"""
X_train = train.values[:,1:]
y_train = train.values[:,:1]
X_test = test.values[:,:]
"""

input_size = X_train.shape[1]

hidden_size = 1000

input_weights = np.random.normal(size=[input_size,hidden_size])
biases = np.random.normal(size=[hidden_size])

def relu(x):
    return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

output_weights = np.dot(pinv2(hidden_nodes(X_train)), y_train)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)
print (prediction)
correct = 0
total = X_test.shape[0]


for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_train[i])
    correct += 1 if predicted == actual else 0

accuracy = correct/total

print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)
