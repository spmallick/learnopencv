#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace std;
using namespace dlib;

void detectFaceDlibHog(frontal_face_detector hogFaceDetector, Mat &frameDlibHog, int inHeight=300, int inWidth=0)
{

    int frameHeight = frameDlibHog.rows;
    int frameWidth = frameDlibHog.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameDlibHogSmall;
    resize(frameDlibHog, frameDlibHogSmall, Size(inWidth, inHeight));

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(frameDlibHogSmall);

    // Detect faces in the image
    std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

    for ( size_t i = 0; i < faceRects.size(); i++ )
    {
      int x1 = (int)(faceRects[i].left() * scaleWidth);
      int y1 = (int)(faceRects[i].top() * scaleHeight);
      int x2 = (int)(faceRects[i].right() * scaleWidth);
      int y2 = (int)(faceRects[i].bottom() * scaleHeight);
      cv::rectangle(frameDlibHog, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}

int main( int argc, const char** argv )
{
  frontal_face_detector hogFaceDetector = get_frontal_face_detector();

  VideoCapture source;
  if (argc == 1)
      source.open(0);
  else
      source.open(argv[1]);
  Mat frame;

  double tt_dlibHog = 0;
  double fpsDlibHog = 0;
  while(1)
  {
      source >> frame;
      if(frame.empty())
          break;

      double t = cv::getTickCount();
      detectFaceDlibHog ( hogFaceDetector, frame );
      tt_dlibHog = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      fpsDlibHog = 1/tt_dlibHog;
      putText(frame, format("DLIB HoG ; FPS = %.2f",fpsDlibHog), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
      imshow( "DLIB - HoG Face Detection", frame );
      int k = waitKey(5);
      if(k == 27)
      {
        destroyAllWindows();
        break;
      }
    }
}
