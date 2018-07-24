import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# open files
file_right = open("../points_mixture_rightside.txt", "r")
file_left = open("../points_mixture_leftside.txt", "r")

# read points
pts_right = file_right.readlines()
pts_left = file_left.readlines()

# copy points to lists
y = []
x = []
labels = []

for index, pt in enumerate(pts_right):
    pt = pt.strip("\n").split()
    
    labels.append(1)
    x.append([float(pt[0]), float(pt[1])])

    pt_left = pts_left[index].strip("\n").split()

    x.append([float(pt_left[0]), float(pt_left[1])])
    labels.append(0)

    # plt.scatter(float(pt[0]), float(pt[1]), c = 'b')

# print(len(x), len(y), len(labels))

x = np.array(x)

# split on 80% - 20% ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, \
test_size = 0.2)

# Training Data Plot
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, s = 30)
plt.title("Training Data")
plt.savefig("linear-data-with-noise-train.png", dpi = 600)
plt.show()

# Testing Data plot
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, s = 30)
plt.title("Testing Data")
plt.savefig("linear-data-with-noise-test.png", dpi = 600)
plt.show()

# make a classifier and fit on training data
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# make predictions
clf_predictions = clf.predict(X_test)

# predictions plot 
plt.scatter(X_test[:, 0], X_test[:, 1], c = clf_predictions, s = 30)

for i, val in enumerate(clf_predictions):
    # change marker
    if(val != y_test[i]):
        marker = 'x'
        plt.scatter(X_test[i, 0], X_test[i, 1], marker = marker, s = 30)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'], )
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, c = 'b', facecolors='none', label = 'Support Vectors')

plt.title("Testing SVM")
plt.savefig("decision-function-on-testing-data.png", dpi=600)
plt.show()

# decision function on training data
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, s = 30)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'], )
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, c = 'b', facecolors='none', label = 'Support Vectors')

plt.title("Decision Function")
plt.savefig("decision-function-on-training-data.png", dpi=600)
plt.show()

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

# make scatter plot for Grid Search
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, s=30)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'], )
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, c = 'b', facecolors='none', label = 'Support Vectors')

plt.title("Decision Function after Gid Search")
plt.legend()
plt.savefig('grid-search-output.png', dpi = 600)
plt.show()
