#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

// Declare variables to store images
Mat gray, cimg, img, edges;

int initThresh;
const int maxThresh = 200;
double p1,p2;

// Vector to store circle points
vector<Vec3f> circles;

void onTrackbarChange( int , void* ) {
  cimg = img.clone();

  p1 = initThresh;
  p2 = initThresh * 0.4;
  
  // Detect circles using HoughCircles transform
  HoughCircles(gray, circles, HOUGH_GRADIENT, 1, cimg.rows/64, p1, p2, 25, 50);

  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // Draw the outer circle
      circle( cimg, center, radius, Scalar(0, 255, 0), 2);
      // Draw the center of the circle
      circle( cimg, center, 2, Scalar(0, 0, 255), 3);
   }

  // Display output image
  imshow( "Image", cimg);

  // Edge image for debugging
  Canny(gray, edges, p1, p2);
  imshow( "Edges", edges);
}


int main(int argc, char** argv){
  const char* file = argv[1];
  img = imread(file, IMREAD_COLOR);

  if(img.empty())
  {
    cout << "Error reading image" << file<< endl;
    return -1;
  }

  // Convert to gray-scale
  cvtColor(img, gray, COLOR_BGR2GRAY);

  // Will hold the results of the detection
  namedWindow("Edges",1);
  namedWindow("Image",1);
 
  initThresh = 105;

  createTrackbar("Threshold", "Image", &initThresh, maxThresh, onTrackbarChange);
  onTrackbarChange(initThresh, 0);

  imshow("Image", img);
  while(true)
  {
    int key;
    key = waitKey( 0  );
    if( (char)key == 27 )
      { break; }
  }

  destroyAllWindows();

}
