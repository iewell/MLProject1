from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import neural_network
import numpy as np;

def load_csv(filename):
    csvfile = open(filename, "r")
    header = csvfile.readline()
    data = np.loadtxt(csvfile, delimiter=';')
    return (header, data[:, :-1], data[:, -1:])

header, data, labels = load_csv("datasets/winequality-red.csv")
data_train = data[:1200]
data_test = data[1201:]
labels_train = np.transpose(labels[:1200])[0]
labels_test = labels[1201:]
print len(data_train), " ", len(data_test)

'''
dt = tree.DecisionTreeClassifier(criterion='entropy', max_features="auto")
dt.fit(data_train, labels_train)
infer = dt.predict(data_test)
'''

dt = neural_network.MLPClassifier()
dt.fit(data_train, labels_train)
infer = dt.predict(data_test)


#dt = neighbors.KNeighborsClassifier(n_neighbors=143)
#dt.fit(data_train, labels_train)
#infer = dt.predict(data_test)

accuracy = metrics.accuracy_score(labels_test, infer)
print "Test accuracy: ", accuracy
#tree.export_graphviz(dt, out_file='tree.dot')
