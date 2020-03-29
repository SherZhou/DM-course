from sklearn import datasets
from decision_tree import DecisionTree

# load data
Iris = datasets.load_iris()
X_train = Iris.data
Y_train = Iris.target
# run
clf = DecisionTree()
clf.fit(X_train, Y_train)
