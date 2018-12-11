#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  Mat im1 = imread("images/K.png",IMREAD_GRAYSCALE); 
  Mat im2 = imread("images/K-rotated.png",IMREAD_GRAYSCALE); 
  Mat im3 = imread("images/A.png",IMREAD_GRAYSCALE); 

  double m1 = matchShapes(im1, im1, CV_CONTOURS_MATCH_I1, 0);
  double m2 = matchShapes(im1, im2, CV_CONTOURS_MATCH_I1, 0);
  double m3 = matchShapes(im1, im3, CV_CONTOURS_MATCH_I1, 0);

  cout << "Shape Distances Between " << endl << "-------------------------" << endl; 
  cout << "K.png and K.png : " << m1 << endl; 
  cout << "K.png and K-transformed.png : " << m2 << endl; 
  cout << "K.png and A.png : " << m3 << endl; 
}