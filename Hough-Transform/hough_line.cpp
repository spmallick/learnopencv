#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// variables to store images
Mat dst, cdstP, gray, src;

int thresh;
const int thresh_max = 100;
double t;

// create a vector to store points of line
vector<Vec4i> linesP;

void on_trackbar( int , void* )
{ 
  cdstP = src.clone();
  
  // apply hough line transform
  HoughLinesP(dst, linesP, 1, CV_PI/180, thresh, 10, 250);

  // draw lines on the detected points
   for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, LINE_AA);
    }
   
   // show the resultant image
   imshow("Result Image", cdstP);
}

int main(int argc, char** argv) {
  const char* file = argv[1];
  // Read image (color mode)
  src = imread(file, 1);

  if(src.empty())
  {
    cout << "Error in reading image" << file<< endl;
    return -1;
  }

  // Convert to gray-scale
  cvtColor(src, gray, COLOR_BGR2GRAY);

  // Detect edges using Canny Edge Detector
  Canny(gray, dst, 50, 200, 3);
  
  // Make a copy of original image
  cdstP = src.clone();

  // Will hold the results of the detection
  namedWindow("Result Image", 1);
 
  // Declare thresh to vary the max_radius of circles to be detected in hough transform
  thresh = 50;

  // Create trackbar to change threshold values
  createTrackbar("threshold", "Result Image", &thresh, thresh_max, on_trackbar);
  on_trackbar(thresh, 0);

  // Show the final image with trackbar
  imshow("Source Image", src);
  while(true)
  {
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 )
      { break; }
  }
}
