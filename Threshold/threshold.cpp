/**
 *  OpenCV Threshold Example
 *   
 *  Copyright 2015 by Satya Mallick <spmallick@gmail.com>
 *  
 **/

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

	// Read image 
	Mat src = imread("threshold.png", IMREAD_GRAYSCALE); 
	Mat dst; 
	
	// Basic threhold example 
	threshold(src,dst,0, 255, THRESH_BINARY); 
	imwrite("opencv-threshold-example.jpg", dst); 

	// Thresholding with maxval set to 128
	threshold(src, dst, 0, 128, THRESH_BINARY); 
	imwrite("opencv-thresh-binary-maxval.jpg", dst); 
	
	// Thresholding with threshold value set 127 
	threshold(src,dst,127,255, THRESH_BINARY); 
	imwrite("opencv-thresh-binary.jpg", dst); 
	
	// Thresholding using THRESH_BINARY_INV 
	threshold(src,dst,127,255, THRESH_BINARY_INV); 
	imwrite("opencv-thresh-binary-inv.jpg", dst); 
	
	// Thresholding using THRESH_TRUNC 
	threshold(src,dst,127,255, THRESH_TRUNC); 
	imwrite("opencv-thresh-trunc.jpg", dst); 

	// Thresholding using THRESH_TOZERO 
	threshold(src,dst,127,255, THRESH_TOZERO); 
	imwrite("opencv-thresh-tozero.jpg", dst); 

	// Thresholding using THRESH_TOZERO_INV 
	threshold(src,dst,127,255, THRESH_TOZERO_INV); 
	imwrite("opencv-thresh-to-zero-inv.jpg", dst); 
} 
