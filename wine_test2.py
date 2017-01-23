from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import neural_network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

from time import time
from scipy.stats import randint as sp_randint

import numpy as np;

def load_csv(filename):
    csvfile = open(filename, "r")
    header = csvfile.readline()
    data = np.loadtxt(csvfile, delimiter=';')
    return  (header, data[:, :-1], data[:, -1:])

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

header, data, labels = load_csv("datasets/winequality-red.csv")
#header, data, labels = load_csv("datasets/bank.csv")

data_train = data[:1200]
data_test = data[1201:]
labels_train = np.transpose(labels[:1200])[0]
labels_test = labels[1201:]
print len(data_train), " ", len(data_test)

'''
#dt = tree.DecisionTreeClassifier(criterion='entropy', max_features="auto")
#dt.fit(data_train, labels_train)
#infer = dt.predict(data_test)
'''

'''
#dt = neighbors.KNeighborsClassifier(n_neighbors=143)
#dt.fit(data_train, labels_train)
#infer = dt.predict(data_test)
'''

'''
dt = ensemble.AdaBoostClassifier()
dt.fit(data_train, labels_train)
infer = dt.predict(data_test)
'''

'''
dt = svm.SVC(kernel="linear", C=0.025)
dt.fit(data_train, labels_train)
infer = dt.predict(data_test)
'''

print "Starting search for SVM model"
dt = svm.SVC()

param_dist = {"degree": sp_randint(2, 5),
              "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
              }

n_iter_search = 15
random_search = RandomizedSearchCV(dt, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=8)

start = time()
random_search.fit(data_train, labels_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#print random_search.cv_results_
report(random_search.cv_results_)
infer = random_search.predict(data_test)
accuracy = metrics.accuracy_score(labels_test, infer)
print "Test accuracy: ", accuracy

print "Starting search for decision tree model"
dt = tree.DecisionTreeClassifier()

param_dist = {"max_depth": sp_randint(2, 30),
              "criterion": ['gini', 'entropy'],
              "max_features": ['sqrt', 'log2', None],
              "min_samples_split": sp_randint(2, 10),
              }

n_iter_search = 5000
random_search = RandomizedSearchCV(dt, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=8)

start = time()
random_search.fit(data_train, labels_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#print random_search.cv_results_
report(random_search.cv_results_)
infer = random_search.predict(data_test)
accuracy = metrics.accuracy_score(labels_test, infer)
print "Test accuracy: ", accuracy
tree.export_graphviz(random_search.best_estimator_, out_file='tree.dot')

print "Starting search for KNN model"
dt = neighbors.KNeighborsClassifier()

param_dist = {"n_neighbors": sp_randint(1, 200),
              "p": sp_randint(1, 2),
              }

n_iter_search = 400
random_search = RandomizedSearchCV(dt, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=8)

start = time()
random_search.fit(data_train, labels_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#print random_search.cv_results_
report(random_search.cv_results_)
infer = random_search.predict(data_test)
accuracy = metrics.accuracy_score(labels_test, infer)
print "Test accuracy: ", accuracy

print "Starting search for neural network model"
dt = neural_network.MLPClassifier(early_stopping=True)

# specify parameters and distributions to sample from
param_dist = {"hidden_layer_sizes": sp_randint(200,1000),
              "activation": ['logistic', 'tanh', 'relu'],
              "solver": ['lbfgs', 'sgd', 'adam'],
              "alpha": [1, 0.1, 0.01, 0.001],
              #"max_iter": sp_randint(50, 500),
              "learning_rate_init": np.logspace(-1,-3,50, endpoint=True)
              }

# run randomized search
n_iter_search = 500
random_search = RandomizedSearchCV(dt, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=8)

start = time()
random_search.fit(data_train, labels_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#print random_search.cv_results_
report(random_search.cv_results_)
infer = random_search.predict(data_test)
accuracy = metrics.accuracy_score(labels_test, infer)
print "Test accuracy: ", accuracy

