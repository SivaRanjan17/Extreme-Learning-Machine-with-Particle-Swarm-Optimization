from numpy import mean, std

from ELM import ELMRegressor, ELMClassifier
from time import time
from sklearn.cluster import k_means
import pandas as pd
#from elm import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
from sklearn.model_selection import train_test_split

def res_dist(x, y, e, n_runs=100, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)

    test_res = []
    train_res = []
    start_time = time()

    for i in range(n_runs):
        e.fit(x_train, y_train)
        train_res.append(e.score(x_train, y_train))
        test_res.append(e.score(x_test, y_test))
        if (i%(n_runs/10) == 0): print ("%d"%i),

    print ("\nTime: %.3f secs" % (time() - start_time))

    print ("Test Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(test_res), mean(test_res), max(test_res), std(test_res)))
    print ("Train Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(train_res), mean(train_res), max(train_res), std(train_res)))
    print
    return (train_res, test_res)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.values[:,1:]
y_train = test.values[:,:1]
X_test = test.values[:,:]

for af in RandomLayer.activation_func_names():
    print (af)
    elmc = ELMClassifier(activation_func=af)
    tr,ts = res_dist(X_train, y_train, elmc, n_runs=200, random_state=0)