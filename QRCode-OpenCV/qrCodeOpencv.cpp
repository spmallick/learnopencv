#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void display(Mat &im, Mat &bbox)
{
  int index = 0;
  int max_index = (bbox.cols*2);
  for(int l = 0 ; l < bbox.cols; l++) {
    line(
        im,
        Point(
          bbox.at<float>(0, index),
          bbox.at<float>(0, index+1)),
        Point(
          bbox.at<float>(0, (index+2) % max_index),
          bbox.at<float>(0, (index+3) % max_index)),
        Scalar(0,255,0),
        2);
    index+=2;
  }
  imshow("Result", im);
}

int main(int argc, char* argv[])
{
  // Read image
  Mat inputImage;
  if(argc>1)
    inputImage = imread(argv[1]);
  else
    inputImage = imread("qrcode-learnopencv.jpg");

  QRCodeDetector qrDecoder;

  Mat bbox, rectifiedImage;

  std::string data = qrDecoder.detectAndDecode(inputImage, bbox, rectifiedImage);
  if(data.length()>0)
  {
    cout << "Decoded Data : " << data << endl;

    display(inputImage, bbox);
    rectifiedImage.convertTo(rectifiedImage, CV_8UC3);
    imshow("Rectified QRCode", rectifiedImage);

    waitKey(0);
  }
  else
    cout << "QR Code not detected" << endl;
}
