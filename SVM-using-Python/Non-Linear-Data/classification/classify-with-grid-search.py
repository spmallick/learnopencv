import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV

# read files
file_right = open("../points_rightside.txt", "r")
file_left = open("../points_leftside.txt", "r")

pts_right = file_right.readlines()
pts_left = file_left.readlines()

# append points to lists
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

x = np.array(x)

# split to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, \
test_size = 0.2, random_state = 101)

# training data plot
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, s = 30)
plt.title("Training Data")
plt.savefig("non-linear-train-data.png", dpi=600)
plt.show()

# make a classifier
clf = svm.SVC(C = 1.0, kernel='rbf')
clf.fit(X_train, y_train)
# make predictions on test data
clf_predictions = clf.predict(X_test)

# decision function on testing data
plt.scatter(X_test[:, 0], X_test[:, 1], c = clf_predictions, s = 30)

for index, prediction in enumerate(clf_predictions):
    if(prediction != y_test[index]):
        # make 'x' for wrong predictions
        marker = 'x'
        plt.scatter(X_test[index, 0], X_test[index, 1], marker = 'x', s = 50)

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

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, s=30)

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
plt.legend()
plt.savefig('decision-function-on-training-data.png', dpi = 600)
plt.show()

# plot for grid search
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.5, 0.01, 0.001, 0.00001]}
grid = GridSearchCV(svm.SVC(), param_grid, verbose=0)
grid.fit(X_train, y_train)
# clf = grid.best_estimator_()
print("Best Parameters:\n", grid.best_params_)
print("Best Estimators:\n", grid.best_estimator_)

clf = svm.SVC(C = grid.best_params_['C'], gamma = grid.best_params_['gamma'])
clf.fit(X_train, y_train)

grid_predictions = clf.predict(X_test)

# print(confusion_matrix(y_test, grid_predictions))

# decision function on testing data
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, s=30)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = grid.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'], )
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, c = 'b', facecolors='none', label = 'Support Vectors')

plt.title("Decision Function after Grid Search")
plt.legend()
plt.savefig('grid-search-decision-function-on-testing-data.png', dpi = 600)
plt.show()
