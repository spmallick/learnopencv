import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np

# open points files
file_right = open("../points_rightside.txt", "r")
file_left = open("../points_leftside.txt", "r")

# read points
pts_right = file_right.readlines()
pts_left = file_left.readlines()

# make lists and append points to the lists
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

# split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, \
test_size = 0.2)

# experiment on changing colors 
# did not work --
colors_train = []
for i, val in enumerate(y_train):
    if(val == 0):
        colors_train.append(0.3)
    else:
        colors_train.append(0.6)

colors_test = []
for i, val in enumerate(y_test):
    if(val == 0):
        colors_test.append(0.3)
    else:
        colors_test.append(0.6)

# training data plot
plt.scatter(X_train[:, 0], X_train[:, 1], c = colors_train, s = 30)
plt.title("Training Data")
plt.savefig("linear-separable-data-train.png", dpi = 600)
plt.show()

# testing data plot
plt.scatter(X_test[:, 0], X_test[:, 1], c = colors_test, s = 30)
plt.title("Testing Data")
plt.savefig("linear-separable-data-test.png", dpi = 600)
plt.show()

# make a classifier
clf = svm.SVC(C = 1.0, kernel='linear')
clf.fit(X_train, y_train)
# make predictions on test data
clf_predictions = clf.predict(X_test)

# experiment with colors
colors = []
for index, prediction in enumerate(clf_predictions):
    if(prediction == 0):
        colors.append(0.6)
    else:
        colors.append(0.3)

# drawing decision function plot
# on testing data

plt.scatter(X_test[:, 0], X_test[:, 1], c = colors, s = 30)

for index, prediction in enumerate(clf_predictions):
    if(prediction == 0):
        color = 0
    else:
        color = 1
    if(prediction != y_test[index]):
        # set marker = 'x' for wrong predictions
        marker = 'x'
        plt.scatter(X_test[index, 0], X_test[index, 1], marker = marker)

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
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, facecolors='none', c = 'b', label='Support Vectors')

plt.legend()
plt.title("Testing SVM")
plt.savefig('decision-boundary-on-testing-data.png', dpi=600)
plt.show()

# make decision function plot
# with training data in it.

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
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, facecolors='none', c = 'b', label='Support Vectors')

plt.legend()
plt.title("Decision Function")
plt.savefig('decision-function-on-training-data.png', dpi=600)
plt.show()
