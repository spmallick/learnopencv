#include "opencv2/opencv.hpp"
#include <iostream>
#include <cstring>

using namespace cv;
using namespace std;
// global variable to keep track of
bool show = false;


// Create a callback for event on trackbars
void onTrackbarActivity(int pos, void* userdata){
	// Just uodate the global variable that there is an event 
	show = true;
	return;
}


int main(int argc, char **argv)
{
	int image_number = 0;
    int nImages = 10;
    if(argc > 1)
        nImages = atoi(argv[1]);
    char filename[20];
    sprintf(filename,"images/rub%02d.jpg",image_number%nImages);

    Mat original = imread(filename);

	// image resize width and height 
	int resizeHeight = 250;
	int resizeWidth = 250;
	Size rsize(resizeHeight,resizeWidth);
	resize(original, original, rsize);

	// position on the screen where the windows start 
	int initialX = 50;
	int	initialY = 50;
	
	// creating windows to display images 
	namedWindow("P-> Previous, N-> Next", WINDOW_AUTOSIZE);
	namedWindow("SelectBGR", WINDOW_AUTOSIZE);
	namedWindow("SelectHSV", WINDOW_AUTOSIZE);
	namedWindow("SelectYCB", WINDOW_AUTOSIZE);
	namedWindow("SelectLAB", WINDOW_AUTOSIZE);
	
	// moving the windows to stack them horizontally 
	moveWindow("P-> Previous, N-> Next", initialX, initialY);
	moveWindow("SelectBGR", initialX + 1 * (resizeWidth + 5), initialY);
	moveWindow("SelectHSV", initialX + 2 * (resizeWidth + 5), initialY);
	moveWindow("SelectYCB", initialX + 3 * (resizeWidth + 5), initialY);
	moveWindow("SelectLAB", initialX + 4 * (resizeWidth + 5), initialY);
	
	// creating trackbars to get values for YCrCb 
	createTrackbar("CrMin", "SelectYCB", 0, 255, onTrackbarActivity);
	createTrackbar("CrMax", "SelectYCB", 0, 255, onTrackbarActivity);
	createTrackbar("CbMin", "SelectYCB", 0, 255, onTrackbarActivity);
	createTrackbar("CbMax", "SelectYCB", 0, 255, onTrackbarActivity);
	createTrackbar("YMin", "SelectYCB", 0, 255, onTrackbarActivity);
	createTrackbar("YMax", "SelectYCB", 0, 255, onTrackbarActivity);

	// creating trackbars to get values for HSV 
	createTrackbar("HMin", "SelectHSV", 0, 180, onTrackbarActivity);
	createTrackbar("HMax", "SelectHSV", 0, 180, onTrackbarActivity);
	createTrackbar("SMin", "SelectHSV", 0, 255, onTrackbarActivity);
	createTrackbar("SMax", "SelectHSV", 0, 255, onTrackbarActivity);
	createTrackbar("VMin", "SelectHSV", 0, 255, onTrackbarActivity);
	createTrackbar("VMax", "SelectHSV", 0, 255, onTrackbarActivity);

	// creating trackbars to get values for BGR 
	createTrackbar("BMin", "SelectBGR", 0, 255, onTrackbarActivity);
	createTrackbar("BMax", "SelectBGR", 0, 255, onTrackbarActivity);
	createTrackbar("GMin", "SelectBGR", 0, 255, onTrackbarActivity);
	createTrackbar("GMax", "SelectBGR", 0, 255, onTrackbarActivity);
	createTrackbar("RMin", "SelectBGR", 0, 255, onTrackbarActivity);
	createTrackbar("RMax", "SelectBGR", 0, 255, onTrackbarActivity);

	// creating trackbars to get values for LAB 
	createTrackbar("LMin", "SelectLAB", 0, 255, onTrackbarActivity);
	createTrackbar("LMax", "SelectLAB", 0, 255, onTrackbarActivity);
	createTrackbar("AMin", "SelectLAB", 0, 255, onTrackbarActivity);
	createTrackbar("AMax", "SelectLAB", 0, 255, onTrackbarActivity);
	createTrackbar("BMin", "SelectLAB", 0, 255, onTrackbarActivity);
	createTrackbar("BMax", "SelectLAB", 0, 255, onTrackbarActivity);

	// show all images initially 
	imshow("SelectHSV", original);
	imshow("SelectYCB", original);
	imshow("SelectLAB", original);
	imshow("SelectBGR", original);
	
	// declare local variables
	int BMin, GMin, RMin;
	int BMax, GMax, RMax;
	Scalar minBGR, maxBGR;

	int HMin, SMin, VMin;
	int HMax, SMax, VMax;
	Scalar minHSV, maxHSV;

	int LMin, aMin, bMin;
	int LMax, aMax, bMax;
	Scalar minLab, maxLab;

	int YMin, CrMin, CbMin;
	int YMax, CrMax, CbMax;
	Scalar minYCrCb, maxYCrCb;

	Mat imageBGR, imageHSV, imageLab, imageYCrCb;
	Mat maskBGR, maskHSV, maskLab, maskYCrCb;
	Mat resultBGR, resultHSV, resultLab, resultYCrCb;

	char k;
	while (1){
		imshow("P-> Previous, N-> Next", original);
		k = waitKey(1) & 0xFF;
		//Check next image in the folder
        if (k =='n')
        {
            image_number++;
            sprintf(filename,"images/rub%02d.jpg",image_number%nImages);
            original = imread(filename);
            resize(original,original,rsize); 
            show = true;
        }
        //Check previous image in he folder
        else if (k =='p')
        {
            image_number--;
            sprintf(filename,"images/rub%02d.jpg",image_number%nImages);
            original = imread(filename);
            resize(original,original,rsize);
            show = true;
        }

        // Close all windows when 'esc' key is pressed		
		if (k == 27){
			break;
		}
		
		if (show) { //If there is any event on the trackbar
			show = false;

            // Get values from the BGR trackbar
			BMin = getTrackbarPos("BMin", "SelectBGR");
			GMin = getTrackbarPos("GMin", "SelectBGR");
			RMin = getTrackbarPos("RMin", "SelectBGR");

			BMax = getTrackbarPos("BMax", "SelectBGR");
			GMax = getTrackbarPos("GMax", "SelectBGR");
			RMax = getTrackbarPos("RMax", "SelectBGR");

			minBGR = Scalar(BMin, GMin, RMin);
			maxBGR = Scalar(BMax, GMax, RMax);

            // Get values from the HSV trackbar
			HMin = getTrackbarPos("HMin", "SelectHSV");
			SMin = getTrackbarPos("SMin", "SelectHSV");
			VMin = getTrackbarPos("VMin", "SelectHSV");

			HMax = getTrackbarPos("HMax", "SelectHSV");
			SMax = getTrackbarPos("SMax", "SelectHSV");
			VMax = getTrackbarPos("VMax", "SelectHSV");

			minHSV = Scalar(HMin, SMin, VMin);
			maxHSV = Scalar(HMax, SMax, VMax);

            // Get values from the LAB trackbar
			LMin = getTrackbarPos("LMin", "SelectLAB");
			aMin = getTrackbarPos("AMin", "SelectLAB");
			bMin = getTrackbarPos("BMin", "SelectLAB");

			LMax = getTrackbarPos("LMax", "SelectLAB");
			aMax = getTrackbarPos("AMax", "SelectLAB");
			bMax = getTrackbarPos("BMax", "SelectLAB");

			minLab = Scalar(LMin, aMin, bMin);
			maxLab = Scalar(LMax, aMax, bMax);

            // Get values from the YCrCb trackbar
			YMin = getTrackbarPos("YMin", "SelectYCB");
			CrMin = getTrackbarPos("CrMin", "SelectYCB");
			CbMin = getTrackbarPos("CbMin", "SelectYCB");

			YMax = getTrackbarPos("YMax", "SelectYCB");
			CrMax = getTrackbarPos("CrMax", "SelectYCB");
			CbMax = getTrackbarPos("CbMax", "SelectYCB");

			minYCrCb = Scalar(YMin, CrMin, CbMin);
			maxYCrCb = Scalar(YMax, CrMax, CbMax);

			// Convert the BGR image to other color spaces
			original.copyTo(imageBGR);
			cvtColor(original, imageHSV, COLOR_BGR2HSV);
			cvtColor(original, imageYCrCb, COLOR_BGR2YCrCb);
			cvtColor(original, imageLab, COLOR_BGR2Lab);

			// Create the mask using the min and max values obtained from trackbar and apply bitwise and operation to get the results
			inRange(imageBGR, minBGR, maxBGR, maskBGR);
			resultBGR = Mat::zeros(original.rows, original.cols, CV_8UC3);
			bitwise_and(original, original, resultBGR, maskBGR);

			inRange(imageHSV, minHSV, maxHSV, maskHSV);
			resultHSV = Mat::zeros(original.rows, original.cols, CV_8UC3);
			bitwise_and(original, original, resultHSV, maskHSV);

			inRange(imageYCrCb, minYCrCb, maxYCrCb, maskYCrCb);
			resultYCrCb = Mat::zeros(original.rows, original.cols, CV_8UC3);
			bitwise_and(original, original, resultYCrCb, maskYCrCb);

			inRange(imageLab, minLab, maxLab, maskLab);
			resultLab = Mat::zeros(original.rows, original.cols, CV_8UC3);
			bitwise_and(original, original, resultLab, maskLab);

			// Show the results
			imshow("SelectBGR", resultBGR);
			imshow("SelectYCB", resultYCrCb);
			imshow("SelectLAB", resultLab);
			imshow("SelectHSV", resultHSV);
		}
	}
	destroyAllWindows();
	return 0;
}