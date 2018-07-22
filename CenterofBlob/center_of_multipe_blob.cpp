#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
 
RNG rng(12345);
 
void find_moments( Mat src );
 
int main(int argc, char** argv)
{
    /// Load source image, convert it to gray
    Mat src, gray;
    src = imread(argv[1], 1 );
    
    cvtColor( src, gray, COLOR_BGR2GRAY );
 
    namedWindow( "Source", WINDOW_AUTOSIZE );
    imshow( "Source", src );
    // call function to find_moments
    find_moments( gray );
 
    waitKey(0);
    return(0);
}
 
void find_moments( Mat gray )
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
 
    /// Detect edges using canny
    Canny( gray, canny_output, 50, 150, 3 );
    // Find contours
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
 
    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }
 
    ///  Get the centroid of figures.
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
    
 
    /// Draw contours
    
    Mat drawing(canny_output.size(), CV_8UC3, Scalar(255,255,255));

    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar(167,151,0);
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        circle( drawing, mc[i], 4, color, -1, 7, 0 );
    }
 
    /// Show the resultant image
    namedWindow( "Contours", WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    waitKey(0);
		
}
