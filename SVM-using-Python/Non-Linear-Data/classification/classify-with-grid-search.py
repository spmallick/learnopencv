import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV

import sys
import os

sys.path.append(os.path.abspath("../../"))
from utils import read_data, plot_data, plot_decision_function

# Read data
x, labels = read_data("../points_rightside.txt", "../points_leftside.txt")

# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.2, random_state=0)

plot_data(X_train, y_train, X_test, y_test)

# make a classifier
clf = svm.SVC(C = 1.0, kernel='rbf')

# Train classifier
clf.fit(X_train, y_train)

# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)

# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.5, 0.01, 0.001, 0.00001]}

# Make grid search classifier
grid = GridSearchCV(svm.SVC(), param_grid, verbose=0)

# Train the classifier
grid.fit(X_train, y_train)

# clf = grid.best_estimator_()
print("Best Parameters:\n", grid.best_params_)
print("Best Estimators:\n", grid.best_estimator_)

# Create classifier with best parameters
clf = svm.SVC(C = grid.best_params_['C'], gamma = grid.best_params_['gamma'])
clf.fit(X_train, y_train)

# Make predictions on unseen test data
grid_predictions = clf.predict(X_test)

# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf)