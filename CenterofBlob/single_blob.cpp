#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Declare Mat type images
    Mat src, gray,thr;
    
    //Load source image, convert it to gray
    src = imread(argv[1], 1 );
 	 
	// convert image to grayscale
	cvtColor( src, gray, COLOR_BGR2GRAY );
	 
	// convert grayscale to binary image
	threshold( gray, thr, 100,255,THRESH_BINARY );
	 
	// find moments of the image
	Moments m = moments(thr,true);
	Point p(m.m10/m.m00, m.m01/m.m00);
	 
	// coordinates of centroid
	cout<< Mat(p)<< endl;
	 
	// show the image with a point mark at the centroid
	circle(src, p, 5, Scalar(128,0,0), -1);
	imshow("Image with center",src);
	waitKey(0);
}
