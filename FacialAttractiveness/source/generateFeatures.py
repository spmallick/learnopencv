import math
import numpy
import itertools

def facialRatio(points):
	x1 = points[0];
	y1 = points[1];
	x2 = points[2];
	y2 = points[3];
	x3 = points[4];
	y3 = points[5];
	x4 = points[6];
	y4 = points[7];

	dist1 = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	dist2 = math.sqrt((x3-x4)**2 + (y3-y4)**2)

	ratio = dist1/dist2

	return ratio


def generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates):
	size = allLandmarkCoordinates.shape
	allFeatures = numpy.zeros((size[0], len(pointIndices1)))
	for x in range(0, size[0]):
		landmarkCoordinates = allLandmarkCoordinates[x, :]
		ratios = [];
		for i in range(0, len(pointIndices1)):
			x1 = landmarkCoordinates[2*(pointIndices1[i]-1)]
			y1 = landmarkCoordinates[2*pointIndices1[i] - 1]
			x2 = landmarkCoordinates[2*(pointIndices2[i]-1)]
			y2 = landmarkCoordinates[2*pointIndices2[i] - 1]

			x3 = landmarkCoordinates[2*(pointIndices3[i]-1)]
			y3 = landmarkCoordinates[2*pointIndices3[i] - 1]
			x4 = landmarkCoordinates[2*(pointIndices4[i]-1)]
			y4 = landmarkCoordinates[2*pointIndices4[i] - 1]

			points = [x1, y1, x2, y2, x3, y3, x4, y4]
			ratios.append(facialRatio(points))
		allFeatures[x, :] = numpy.asarray(ratios)
	return allFeatures


def generateAllFeatures(allLandmarkCoordinates):
	a = [18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
	combinations = itertools.combinations(a, 4)
	i = 0
	pointIndices1 = [];
	pointIndices2 = [];
	pointIndices3 = [];
	pointIndices4 = [];

	for combination in combinations:
		pointIndices1.append(combination[0])
		pointIndices2.append(combination[1])
		pointIndices3.append(combination[2])
		pointIndices4.append(combination[3])
		i = i+1
		pointIndices1.append(combination[0])
		pointIndices2.append(combination[2])
		pointIndices3.append(combination[1])
		pointIndices4.append(combination[3])
		i = i+1
		pointIndices1.append(combination[0])
		pointIndices2.append(combination[3])
		pointIndices3.append(combination[1])
		pointIndices4.append(combination[2])
		i = i+1

	return generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates)

landmarks = numpy.loadtxt('landmarks.txt', delimiter=',', usecols=range(136))

featuresALL = generateAllFeatures(landmarks)
numpy.savetxt('features_ALL.txt', featuresALL, delimiter=',', fmt = '%.04f')

#pointIndices1 = [20, 20, 45, 45]
#pointIndices2 = [58, 9, 58, 58]
#pointIndices3 = [5, 7, 5, 32]
#pointIndices4 = [13, 13, 11, 36]
#features = generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, landmarks)

