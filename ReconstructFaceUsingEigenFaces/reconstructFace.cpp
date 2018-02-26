#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;


// Matrices for average (mean) and eigenvectors
Mat averageFace;
Mat output;
vector<Mat> eigenFaces;
Mat imVector, meanVector, eigenVectors, im, display;

// Display result
// Left = Original Image
// Right = Reconstructed Face
void displayResult( Mat &left, Mat &right)
{
	hconcat(left,right, display); 	
	resize(display, display, Size(), 4, 4);
	imshow("Result", display);
}

// Recontruct face using mean face and EigenFaces
void reconstructFace(int sliderVal, void*)
{
	// Start with the mean / average face
	Mat output = averageFace.clone();
	for (int i = 0;  i < sliderVal; i++)
	{
		// The weight is the dot product of the mean subtracted
		// image vector with the EigenVector
		double weight = imVector.dot(eigenVectors.row(i)); 

		// Add weighted EigenFace to the output
		output = output + eigenFaces[i] * weight; 
	}

	displayResult(im, output);
}
	

int main(int argc, char **argv)
{

	// Read model file
	string modelFile("pcaParams.yml");
	cout << "Reading model file " << modelFile << " ... " ; 

	FileStorage file(modelFile, FileStorage::READ);
	
	// Extract mean vector
	meanVector = file["mean"].mat();

	// Extract Eigen Vectors
	eigenVectors = file["eigenVectors"].mat();

	// Extract size of the images used in training.
	Mat szMat = file["size"].mat();
	Size sz = Size(szMat.at<double>(1,0),szMat.at<double>(0,0));

	// Extract maximum number of EigenVectors. 
	// This is the max(numImagesUsedInTraining, w * h * 3)
	// where w = width, h = height of the training images. 
	int numEigenFaces = eigenVectors.size().height; 
	cout <<  "DONE" << endl; 

	cout << "Extracting mean face and eigen faces ... "; 
	// Extract mean vector and reshape it to obtain average face
	averageFace = meanVector.reshape(3,sz.height);
	
	// Reshape Eigenvectors to obtain EigenFaces
	for(int i = 0; i < numEigenFaces; i++)
	{
			Mat row = eigenVectors.row(i); 
			Mat eigenFace = row.reshape(3,sz.height);
			eigenFaces.push_back(eigenFace);
	}
	cout << "DONE" << endl; 

	// Read new test image. This image was not used in traning. 
	string imageFilename("test/satya1.jpg");
	cout << "Read image " << imageFilename << " and vectorize ... ";
	im = imread(imageFilename);
	im.convertTo(im, CV_32FC3, 1/255.0);
	
	// Reshape image to one long vector and subtract the mean vector
	imVector = im.clone(); 
	imVector = imVector.reshape(1, 1) - meanVector; 
	cout << "DONE" << endl; 


	// Show mean face first
	output = averageFace.clone(); 

	cout << "Usage:" << endl 
	<< "\tChange the slider to change the number of EigenFaces" << endl
	<< "\tHit ESC to terminate program." << endl;
	
	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	int sliderValue; 

	// Changing the slider value changes the number of EigenVectors
	// used in reconstructFace.
	createTrackbar( "No. of EigenFaces", "Result", &sliderValue, numEigenFaces, reconstructFace);
	
	// Display original image and the reconstructed image size by side
	displayResult(im, output);
	

	waitKey(0);
	destroyAllWindows(); 
}


