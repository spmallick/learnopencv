#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  Mat im1 = imread("images/S0.png",IMREAD_GRAYSCALE);
  Mat im2 = imread("images/K0.png",IMREAD_GRAYSCALE);
  Mat im3 = imread("images/S4.png",IMREAD_GRAYSCALE);

  double m1 = matchShapes(im1, im1, CONTOURS_MATCH_I2, 0);
  double m2 = matchShapes(im1, im2, CONTOURS_MATCH_I2, 0);
  double m3 = matchShapes(im1, im3, CONTOURS_MATCH_I2, 0);

  cout << "Shape Distances Between " << endl << "-------------------------" << endl;
  cout << "S0.png and S0.png : " << m1 << endl;
  cout << "S0.png and K0.png : " << m2 << endl;
  cout << "S0.png and S4.png : " << m3 << endl;
}
