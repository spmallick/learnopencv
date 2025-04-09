#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // Print OpenCV Version
    cout << CV_VERSION << endl;
    
    // Read image
    Mat image = imread("boy.jpg", 1);
    
    // Convert image to grayscale
    cvtColor(image, image, COLOR_BGR2GRAY);
    
    // Display image
    imshow("Display", image);
    waitKey(0);
    
    // Save grayscale image
    imwrite("boyGray.jpg",image);
    
    return 0;
}
