import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from numpy import argmax, array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2

# define example

# define dataset

train = pd.read_csv("kdd_train.csv")

print(train.shape)

for i in ['protocol_type', 'service', "flag", "labels"]:
	values = array(train[i])
	#print(values)
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	#print(integer_encoded)
	train[i] = integer_encoded
	#print (train[i])

X = train.values[:, :41]
#print (X)
y = train.values[:, 41]
#print (y)

# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=20)
# fit RFE
rfe.fit(X, y)
# summarize all features
rfe_X = pd.DataFrame()
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
	#print (train.columns[i])
	col_name = train[train.columns[i]]
	if rfe.support_[i]:
		rfe_X = pd.concat([rfe_X, col_name], axis = 1)

print (rfe_X.shape, rfe_X.head())

#ELM
input_size = rfe_X.shape[1]

hidden_size = 500

input_weights = np.random.normal(size=[input_size, hidden_size])
biases = np.random.normal(size=[hidden_size])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = sigmoid(G)
    return H

output_weights = np.dot(pinv2(hidden_nodes(rfe_X)), y)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(rfe_X)
print (prediction, y)

correct = 0
total = rfe_X.shape[0]


for i in range(total):
    predicted = round(prediction[i])
    actual = y[i]
    #print(prediction[i], predicted, y[i], actual)
    correct += 1 if predicted == actual else 0

accuracy = correct/total

print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)
