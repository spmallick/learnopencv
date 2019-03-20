#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// variables to store images
Mat dst, cimg, gray, img, edges;

int initThresh;
const int maxThresh = 1000;
double th1,th2;

// create a vector to store points of line
vector<Vec4i> lines;

void onTrackbarChange( int , void* )
{ 
  cimg = img.clone();
  dst = img.clone();

  th1 = initThresh;
  th2 = th1 * 0.4;

  Canny(img,edges,th1,th2);
  
  // apply hough line transform
  HoughLinesP(edges, lines, 2, CV_PI/180, 50, 10, 100);

  // draw lines on the detected points
   for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, LINE_AA);
    }
   
   // show the resultant image
   imshow("Result Image", dst);
   imshow("Edges", edges);
}

int main(int argc, char** argv) {
  const char* file = argv[1];
  // Read image (color mode)
  img = imread(file, 1);
  dst = img.clone();

  if(img.empty())
  {
    cout << "Error in reading image" << file<< endl;
    return -1;
  }

  // Convert to gray-scale
  cvtColor(img, gray, COLOR_BGR2GRAY);

  // Detect edges using Canny Edge Detector
  // Canny(gray, dst, 50, 200, 3);
  
  // Make a copy of original image
  // cimg = img.clone();

  // Will hold the results of the detection
  namedWindow("Edges",1);
  namedWindow("Result Image", 1);
 
  // Declare thresh to vary the max_radius of circles to be detected in hough transform
  initThresh = 500;

  // Create trackbar to change threshold values
  createTrackbar("threshold", "Result Image", &initThresh, maxThresh, onTrackbarChange);
  onTrackbarChange(initThresh, 0);

  while(true)
  {
    int key;
    key = waitKey( 1 );
    if( (char)key == 27 )
      { break; }
  }

  destroyAllWindows();
}
