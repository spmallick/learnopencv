#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
		
	Mat testImage = imread("boat.jpg", 0);

	Mat dst;
	double thresh = 0;
	double maxValue = 255;

	long double thres = cv::threshold(testImage,dst, thresh, maxValue, THRESH_OTSU);

	cout << "Otsu Threshold : " << thres <<endl;
	
	return 0;
	
}
