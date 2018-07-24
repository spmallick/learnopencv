import numpy as np 
import matplotlib.pyplot as plt 

data_colors = [(1, 0, 0), (0, 0, 1)]

def read_points_file(filename):
  pts = []
  with open(filename, "r") as f:
    for pt in f:
      pt = pt.strip("\n").split()
      pts.append([float(pt[0]), float(pt[1])])
  return pts

def read_data(classAFile, classBFile):
  pts_right = read_points_file(classAFile)
  pts_left = read_points_file(classBFile)

  x = pts_right + pts_left
  labels = [1] * len(pts_right) + [0] * len(pts_left)
  x = np.array(x)
  return (x, labels)

def plot_data(X_train, y_train, X_test, y_test):
  
  colors_train = get_colors(y_train)
  colors_test = get_colors(y_test)

  # training data plot
  plt.subplot(121)
  plt.axis('equal')
  plt.axis('off')
  plt.scatter(X_train[:, 0], X_train[:, 1], c = colors_train, s = 10, edgecolors=colors_train)
  plt.title("Training Data")

  # testing data plot
  plt.subplot(122)
  plt.axis('equal')
  plt.axis('off')
  plt.scatter(X_test[:, 0], X_test[:, 1], c = colors_test, s = 10, edgecolors=colors_test)
  plt.title("Test Data")
  
  plt.savefig("linear-separable-training-test-data.png", dpi = 600)
  plt.show()

def get_colors(y):
  return [data_colors[item] for item in y]

def plot_decision_function(X_train, y_train, X_test, y_test, clf):
  plt.subplot(121)
  plt.title("Training data")
  plot_decision_function_helper(X_train, y_train, clf)
  plt.subplot(122)
  plt.title("Test data")
  plot_decision_function_helper(X_test, y_test, clf)
  plt.savefig('decision-function-on-training-and-test-data.png', dpi=600)
  plt.show()


def plot_decision_function_helper(X, y, clf):
  
  colors = get_colors(y)
  plt.axis('equal')
  plt.axis('off')

  plt.scatter(X[:, 0], X[:, 1], c = colors, s = 10, edgecolors=colors)
  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()

  # Create grid to evaluate model
  xx = np.linspace(xlim[0], xlim[1], 30)
  yy = np.linspace(ylim[0], ylim[1], 30)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  Z = clf.decision_function(xy).reshape(XX.shape)

  # plot decision boundary and margins
  ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
             linestyles=['--', '-', '--'])
  # plot support vectors
  ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 10,
             linewidth=1, facecolors='k', c = 'k', label='Support Vectors')
 
  plt.legend(fontsize='small')
  
