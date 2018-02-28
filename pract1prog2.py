from sklearn.datasets import load_iris,make_classification
from sklearn import neighbors
irisData = load_iris()
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import random # to generate random numbers

X = irisData.data
Y = irisData.target
kf = KFold(n_splits=3, shuffle=False)
scores = []
for k in range(1,30):
    score = 0
    clf = neighbors.KNeighborsClassifier(k)
    for learn,test in kf.split(X):
        X_train = X[learn]
        Y_train = Y[learn]
        clf.fit(X_train, Y_train)
        X_test = X[test]
        Y_test = Y[test]
        score = score + clf.score(X_test, Y_test)
    scores.append(score)
    print(scores)
print("best k:", scores.index(max(scores))+1)
#BEST K = 13 with n_splits=10, shuffle=True
#BEST k =1 with n_splits=3, shuffle=False