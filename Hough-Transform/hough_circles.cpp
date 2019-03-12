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
Mat dst, gray_src, cdstP, src;

int thresh;
const int thresh_max = 100;
double t;

// Vector to store circle points
vector<Vec3f> circles;

void on_trackbar( int , void* ) {
  cdstP = gray_src.clone();
  
  // Apply hough transform
  HoughCircles(cdstP, circles, HOUGH_GRADIENT, 1, cdstP.rows/64, 200, 10, 1, thresh);
  
  cdstP = src.clone();

  // Draw circle on points
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      circle( cdstP, center, radius, Scalar(255, 255, 255), 2, 8, 0 );
   }

    cv::putText(cdstP, to_string(circles.size()), cv::Point(280, 60), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 255), 4, cv::LINE_AA);
    imshow( "Output-Image", cdstP);
}


int main(int argc, char** argv){
  const char* file = argv[1];
  src = imread(file, IMREAD_COLOR);
  cv::resize(src, src, cv::Size(400, 400));

  if(src.empty())
  {
    cout << "Error reading image" << file<< endl;
    return -1;
  }

  // Remove noise using medianBlur
  medianBlur(src, src, 3);

  // Convert to gray-scale
  cvtColor(src, gray_src, COLOR_BGR2GRAY);

   // Will hold the results of the detection
  namedWindow("Output-Image", 1);
 
  thresh = 10;

  createTrackbar("threshold", "Output-Image", &thresh, thresh_max, on_trackbar);
  on_trackbar(thresh, 0);

  imshow("source", src);
  while(true)
  {
    int c;
    c = waitKey( 0  );
    if( (char)c == 27 )
      { break; }
  }

}
