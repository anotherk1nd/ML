Thursday night,

Before 4am Friday Morning

help(len)
Help on built-in function len in module builtins:
len(obj, /)
#    Return the number of items in a container.

# [2]
print(irisData.target_names[0])
# Prints the first entry of datafield target_names of the irisData
setosa


# [3]

print(irisData.target_names[-1+len(irisData.target_names)])
# Prints last entry of datafield target_names of the irisData, -1 to account for zero indexing
virginica

# [4]
print(irisData.data.shape)
#prints the shape of the dataformat, dimensions of matrix
(150,4)

# [5]
print(irisData.data.shape[0])
#Prints first column of the shape, namely number of rows
150

# [6]
print(irisData.data[0])
#Prints 1st entry of data
[5.1 3.5 1.4 0.2]

# [7], #[8]
plt.scatter(X[:,x], X[:,y], c=Y)
plt.show()
#Plots scatter graph using the data of all rows, with column 1 and column 2 of the data as x and y

![](/home/franky/Documents/Shared Documents/MLDM/Semester 2/ML/num8.png) 

#[9]
plt.ylabel(irisData.feature_names[y])
#Labels the yaxis with the feature name provided in irisData

#[10]
print(Y==0)
#Checks whether Y==0 in all cases, returning TRUE or FALSE
[ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False]


#[11]
print(X[Y==0])
#Returns values of X where Y equals 0. Must be same dimensions
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]
 [5.  3.4 1.5 0.2]
 [4.4 2.9 1.4 0.2]
 [4.9 3.1 1.5 0.1]
 [5.4 3.7 1.5 0.2]
 [4.8 3.4 1.6 0.2]
 [4.8 3.  1.4 0.1]
 [4.3 3.  1.1 0.1]
 [5.8 4.  1.2 0.2]
 [5.7 4.4 1.5 0.4]
 [5.4 3.9 1.3 0.4]
 [5.1 3.5 1.4 0.3]
 [5.7 3.8 1.7 0.3]
 [5.1 3.8 1.5 0.3]
 [5.4 3.4 1.7 0.2]
 [5.1 3.7 1.5 0.4]
 [4.6 3.6 1.  0.2]
 [5.1 3.3 1.7 0.5]
 [4.8 3.4 1.9 0.2]
 [5.  3.  1.6 0.2]
 [5.  3.4 1.6 0.4]
 [5.2 3.5 1.5 0.2]
 [5.2 3.4 1.4 0.2]
 [4.7 3.2 1.6 0.2]
 [4.8 3.1 1.6 0.2]
 [5.4 3.4 1.5 0.4]
 [5.2 4.1 1.5 0.1]
 [5.5 4.2 1.4 0.2]
 [4.9 3.1 1.5 0.1]
 [5.  3.2 1.2 0.2]
 [5.5 3.5 1.3 0.2]
 [4.9 3.1 1.5 0.1]
 [4.4 3.  1.3 0.2]
 [5.1 3.4 1.5 0.2]
 [5.  3.5 1.3 0.3]
 [4.5 2.3 1.3 0.3]
 [4.4 3.2 1.3 0.2]
 [5.  3.5 1.6 0.6]
 [5.1 3.8 1.9 0.4]
 [4.8 3.  1.4 0.3]
 [5.1 3.8 1.6 0.2]
 [4.6 3.2 1.4 0.2]
 [5.3 3.7 1.5 0.2]
 [5.  3.3 1.4 0.2]]
 
 #12,13
 plt.scatter(X[Y==0][:, x], X[Y==0][:, y],
c="red", label=irisData.target_names[0])
plt.scatter(X[Y==1][:, x], X[Y==1][:, y],
c="green", label=irisData.target_names[1])
plt.scatter(X[Y==2][:, x], X[Y==2][:, y],
c="blue", label=irisData.target_names[2])
plt.legend()
plt.show()

#Plots different classes of dataset with different clours on same plot, and provides a legend using the target names

#We make our own classification dataset, The command returns an array X of shape [n_samples,
n_features] which contains the generated samples and an array Y of shape [n_samples] which contains
the integer labels for the class membership of each sample.
make_classification(n_samples=25, n_features=4, n_informative=2, n_redundant=2, n_classes=2)

Exercise 3

#[1]
clf.fit(X, Y)
#Trains a classifier on the data using features X and labels Y

#[2]
print(clf.predict([[ 5.4, 3.2, 1.6, 0.4]])) # [2]
prints the predicted class of a new sample
[1]

# [3]
print(clf.predict_proba([[ 5.4, 3.2, 1.6, 0.4]]))
prints the probability that it belongs to each class
[[0.4 0.6]]

# [4]
print(clf.score(X,Y))
prints the results of the predicting using the trained classifier
0.76

# [5]
Z = clf.predict(X)
ASsociates the predicted classes on the dataset X to variable Z

# [6]
print(X[Z!=Y])
Prints the feature sets of examples that were predicted incorrectly

#[7]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
test_size=0.3,random_state=random.seed())
Splits the dataset into training and test set, in proportions 0.7 and 0.3, using a seed provided by random

#[8]
print(X_train.shape)
Prints the shape of the training set features, we see it it has been split proportionately and has 4 features
(17, 4)

#[9]
print(X_train[Y_train==0].shape)
Prints the shape of the array returned when looking only at training set samples with classification 0
(7, 4)

#[10]
from sklearn.metrics import confusion_matrix
imports the confusion_matrix function

#[11]
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
Prints the confusion matrix, with layout that gives # correctly classified examples in each class, and the false positives and negatives
[[0 6]
 [0 2]]
 
#[12]
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
for learn,test in kf.split(X):
print("app : ", learn, " test ", test)
 When shuffle=false is used, ...
 
#[13]
