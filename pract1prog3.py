from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
print(clf.predict([iris.data[50,:]]))
print(clf.score(iris.data, iris.target))
#tree.export_graphviz(clf, out_file='tree.dot')
