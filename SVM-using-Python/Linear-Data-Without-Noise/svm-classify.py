import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import sys 
import os 

sys.path.append(os.path.abspath("../"))
from utils import read_data, plot_data, plot_decision_function

# Read data
x, labels = read_data("points_class_0.txt", "points_class_1.txt")

# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.2, random_state=0)

print("Displaying data. Close window to continue.")  
# Plot traning and test data
plot_data(X_train, y_train, X_test, y_test)


print('Training Linear SVM')
# Create a linear SVM classifier 
clf = svm.SVC(kernel='linear')

# Train classifier 
clf.fit(X_train, y_train)

print("Displaying decision function. Close window to continue.")  
# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))

