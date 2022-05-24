/*
 * Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
 * All rights reserved. No warranty, explicit or implicit, provided.
 */

#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> tri1, vector<Point2f> tri2)
{
    // Find bounding rectangle for each triangle
    Rect r1 = boundingRect(tri1);
    Rect r2 = boundingRect(tri2);
    
    // Offset points by left top corner of the respective rectangles
    vector<Point2f> tri1Cropped, tri2Cropped;
    vector<Point> tri2CroppedInt;
    for(int i = 0; i < 3; i++)
    {
        tri1Cropped.push_back( Point2f( tri1[i].x - r1.x, tri1[i].y -  r1.y) );
        tri2Cropped.push_back( Point2f( tri2[i].x - r2.x, tri2[i].y - r2.y) );

        // fillConvexPoly needs a vector of Point and not Point2f
        tri2CroppedInt.push_back( Point((int)(tri2[i].x - r2.x), (int)(tri2[i].y - r2.y)) );

    }

    // Apply warpImage to small rectangular patches
    Mat img1Cropped;
    img1(r1).copyTo(img1Cropped);

    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( tri1Cropped, tri2Cropped );
    
    // Apply the Affine Transform just found to the src image
    Mat img2Cropped = Mat::zeros(r2.height, r2.width, img1Cropped.type());
    warpAffine( img1Cropped, img2Cropped, warpMat, img2Cropped.size(), INTER_LINEAR, BORDER_REFLECT_101);

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, tri2CroppedInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Copy triangular region of the rectangular patch to the output image
    multiply(img2Cropped,mask, img2Cropped);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Cropped;
    
}

int main( int argc, char** argv)
{
    // Read input image and convert to float
    Mat imgIn = imread("robot.jpg");
    imgIn.convertTo(imgIn, CV_32FC3, 1/255.0);

    // Output image is set to white
    Mat imgOut = Mat::ones(imgIn.size(), imgIn.type());
    imgOut = Scalar(1.0,1.0,1.0);
    
    // Input triangle
    vector <Point2f> triIn;
    triIn.push_back(Point2f(360,200));
    triIn.push_back(Point2d(60,250));
    triIn.push_back(Point2f(450,400));
    
    // Output triangle
    vector <Point2f> triOut;
    triOut.push_back(Point2f(400,200));
    triOut.push_back(Point2f(160,270));
    triOut.push_back(Point2f(400,400));
    
    // Warp all pixels inside input triangle to output triangle
    warpTriangle(imgIn, imgOut, triIn, triOut);
    
    // Draw triangle on the input and output image.
    
    // Convert back to uint because OpenCV antialiasing
    // does not work on image of type CV_32FC3
    
    imgIn.convertTo(imgIn, CV_8UC3, 255.0);
    imgOut.convertTo(imgOut, CV_8UC3, 255.0);
    
    // Draw triangle using this color
    Scalar color = Scalar(255, 150, 0);
    
    // cv::polylines needs vector of type Point and not Point2f
    vector <Point> triInInt, triOutInt;
    for(int i=0; i < 3; i++)
    {
        triInInt.push_back(Point(triIn[i].x,triIn[i].y));
        triOutInt.push_back(Point(triOut[i].x,triOut[i].y));
    }
    
    // Draw triangles in input and output images
    polylines(imgIn, triInInt, true, color, 2, 16);
    polylines(imgOut, triOutInt, true, color, 2, 16);
    
    imshow("Input", imgIn);
    imshow("Output", imgOut);
    waitKey(0);
    
    return 0;
}
