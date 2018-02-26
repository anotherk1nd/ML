from sklearn.datasets import load_iris,make_classification
from sklearn import neighbors
irisData = load_iris()
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random # to generate random numbers

print(irisData.data)
print(irisData.target)
print(irisData.target_names)
print(irisData.feature_names)
print(irisData.DESCR)

print(len(irisData.data))
help(len)

print('number2')
print(irisData.target_names[0])
# [2]

#print(irisData.target_names[2])
#print(irisData.target_names[-1])
print('number3')
print(irisData.target_names[-1+len(irisData.target_names)])

print('number4')
print(irisData.data.shape)

print('number5')
print(irisData.data.shape[0])

print('number6')
print(irisData.data[0])

X = irisData.data
Y = irisData.target
x = 0
y = 1

plt.scatter(X[:,x], X[:,y], c=Y)


plt.ylabel(irisData.feature_names[y])
#plt.show()

print
print(Y==0)
print(X[Y==0])

#12
plt.scatter(X[Y==0][:, x], X[Y==0][:, y],
c="red", label=irisData.target_names[0])
plt.scatter(X[Y==1][:, x], X[Y==1][:, y],
c="green", label=irisData.target_names[1])
plt.scatter(X[Y==2][:, x], X[Y==2][:, y],
c="blue", label=irisData.target_names[2])
plt.legend()
#plt.show()

ds = make_classification(n_samples=25, n_features=4, n_informative=2, n_redundant=2, n_classes=2)
X = ds[0]
Y = ds[1]
#print('happy',X,Y)
nb_neighb = 15
clf = neighbors.KNeighborsClassifier(nb_neighb)
clf.fit(X, Y)
print('Part3')
print(clf.predict([[ 5.4, 3.2, 1.6, 0.4]])) # [2]
print(clf.predict_proba([[ 5.4, 3.2, 1.6, 0.4]])) # [3]
print(clf.score(X,Y)) #[4]
Z = clf.predict(X)# [5]
print(X[Z!=Y])# [6]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=random.seed()) #[7]
print('#8')
print(X_train.shape)

print(X_train[Y_train==0].shape)

clf = clf.fit(X_train, Y_train)
Y_pred =clf.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
for learn,test in kf.split(X): # We learn on app and test on test
    print("app : ", learn, "test ", test)

print("Now with shuffle=False")
kff = KFold(n_splits=10, shuffle=False)
for learn,test in kf.split(X):
    print("app : ", learn, "test ", test)


