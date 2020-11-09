# Classification using SVM

## Import required modules
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

## Creating a sample dataset
X,Y = make_classification(n_samples=1000,n_features=2,n_informative=1,\
n_clusters_per_class=1,n_redundant=0)

## Training and testing split
# dividing data to train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

## Normalize data
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().transform(X_test)

## Training the SVC model
# make a SVC classifier
clf = svm.SVC()
# fit the training data using classifier
clf.fit(X_train, y_train)

## Predicting the trained model on test data
clf_predictions = clf.predict(X_test)
