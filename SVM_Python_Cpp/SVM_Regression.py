# Regression using SVM

## Import required modules
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

## Creating a sample dataset
# create 1000 samples (2 features) 
X, y = make_regression(n_samples = 1000, n_features = 2, n_informative = 2)

## Training and testing split
# dividing data to train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

## Normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

## Training the SVR model
# make a SVR regressor
reg = svm.SVR()
# fit the training data using regressor
reg.fit(X_train, y_train)

## Predicting the trained model on test data
reg_predictions = reg.predict(X_test)
