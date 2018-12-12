#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  bool showLogTransformedHuMoments = true; 

  for (int i = 1; i < argc; i++)
  {
    // Obtain filename from command line argument
    string filename(argv[i]); 

    // Read Image
    Mat im = imread(filename,IMREAD_GRAYSCALE); 
    
    // Threshold image
    threshold(im, im, 128, 255, THRESH_BINARY);
  
    // Calculate Moments
    Moments moment = moments(im, false);

    // Calculate Hu Moments
    double huMoments[7];
    HuMoments(moment, huMoments);

    // Print Hu Moments
    cout << filename << ": "; 
    
    for(int i = 0; i < 7; i++)
    {
      if(showLogTransformedHuMoments)
      {
        // Log transform Hu Moments to make squash the range
        cout << -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])) << " ";  
      }
      else 
      {
        // Hu Moments without log transform. 
        cout << huMoments[i] << " ";  
      }
      
    }
    // One row per file
    cout << endl; 

  }
  


}