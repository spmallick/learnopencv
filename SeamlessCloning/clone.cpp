/**
 * OpenCV seamlessCloning Example
 *
 * Copyright 2015 by Satya Mallick <spmallick@gmail.com>
 *
 */

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // Read images : src image will be cloned into dst
    Mat src = imread("images/airplane.jpg");
    Mat dst = imread("images/sky.jpg");
    

    // Create a rough mask around the airplane.
    Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
    
    // Define the mask as a closed polygon
    Point poly[1][7];
    poly[0][0] = Point(4, 80);
    poly[0][1] = Point(30, 54);
    poly[0][2] = Point(151,63);
    poly[0][3] = Point(254,37);
    poly[0][4] = Point(298,90);
    poly[0][5] = Point(272,134);
    poly[0][6] = Point(43,122);
    
    const Point* polygons[1] = { poly[0] };
    int num_points[] = { 7 };
    
    // Create mask by filling the polygon
    fillPoly(src_mask, polygons, num_points, 1, Scalar(255,255,255));
    
    // The location of the center of the src in the dst
    Point center(800,100);
    
    // Seamlessly clone src into dst and put the results in output
    Mat output;
    seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);
    
    // Write result
    imwrite("images/opencv-seamless-cloning-example.jpg", output);
    
}
