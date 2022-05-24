import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import decomposition

features = np.loadtxt('features_ALL.txt', delimiter=',')
#features = preprocessing.scale(features)
features_train = features[0:-50]
features_test = features[-50:]

pca = decomposition.PCA(n_components=20)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

ratings = np.loadtxt('labels.txt', delimiter=',')
#ratings = preprocessing.scale(ratings)
ratings_train = ratings[0:-50]
ratings_test = ratings[-50:]

regr = linear_model.LinearRegression()
regr.fit(features_train, ratings_train)
ratings_predict = regr.predict(features_test)
corr = np.corrcoef(ratings_predict, ratings_test)[0, 1]
print corr

residue = np.mean((ratings_predict - ratings_test) ** 2)
print residue

rangeArray = np.arange(1, 51)
plt.plot(rangeArray, ratings_test, 'r', rangeArray, ratings_predict, 'b')
plt.show()
