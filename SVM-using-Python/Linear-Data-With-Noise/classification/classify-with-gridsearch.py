import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import sys
import os

sys.path.append(os.path.abspath("../../"))
from utils import read_data, plot_data, plot_decision_function

# Read data
x, labels = read_data("../points_rightside.txt", "../points_leftside.txt")

# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.2, random_state=0)

plot_data(X_train, y_train, X_test, y_test)

# make a classifier and fit on training data
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# make predictions
clf_predictions = clf.predict(X_test)

# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf)

# GRID SEARCH
# parameters grid for grid search
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.5, 0.01, 0.001, 0.00001]}

# apply grid search
grid = GridSearchCV(svm.SVC(), param_grid, verbose=0)

grid.fit(X_train, y_train)

# print best parameters
print("Best Parameters:\n",grid.best_params_)

print("Best Estimator:\n", grid.best_estimator_)

# make a classifier on based on best parameters
# OR
# clf = grid.best_estimator_
clf = svm.SVC(C = grid.best_params_['C'], gamma = grid.best_params_['gamma'])
clf.fit(X_train, y_train)

# make predictions
grid_predictions = clf.predict(X_test)

# print(confusion_matrix(y_test, grid_predictions))
# print(classification_report(y_test, grid_predictions))

# plot decision function on training and testing data
plot_decision_function(X_train, y_train, X_test, y_test, clf)