/**
 * OpenCV Colormap  Example
 *
 * Copyright 2015 by Satya Mallick <spmallick@learnopencv.com>
 *
 */

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

string colormap_name(int id)
{
    switch(id){
        case COLORMAP_AUTUMN :
            return "COLORMAP_AUTUMN";
        case COLORMAP_BONE :
            return "COLORMAP_BONE";
        case COLORMAP_JET :
            return "COLORMAP_JET";
        case COLORMAP_WINTER :
            return "COLORMAP_WINTER";
        case COLORMAP_RAINBOW :
            return "COLORMAP_RAINBOW";
        case COLORMAP_OCEAN :
            return "COLORMAP_OCEAN";
        case COLORMAP_SUMMER:
            return "COLORMAP_SUMMER";
        case COLORMAP_SPRING :
            return "COLORMAP_SPRING";
        case COLORMAP_COOL :
            return "COLORMAP_COOL";
        case COLORMAP_HSV :
            return "COLORMAP_HSV";
        case COLORMAP_PINK :
            return "COLORMAP_PINK";
        case COLORMAP_HOT :
            return "COLORMAP_HOT";
        
    }
    
    return "NONE";
}

int main( int argc, char** argv )
{
    // Read 8-bit grayscale image
    Mat im = imread("pluto.jpg", IMREAD_GRAYSCALE);

    Mat im_out = Mat::zeros(600, 800, CV_8UC3);

    for (int i=0; i < 4; i++){
        for(int j=0; j < 3; j++){
            int k = i + j * 4;
            Mat im_color = im_out(Rect(i * 200, j * 200, 200, 200));
            applyColorMap(im, im_color, k);
            putText(im_color, colormap_name(k), Point(30, 180), CV_FONT_HERSHEY_DUPLEX, 0.5, Scalar::all(255), 1, CV_AA);
        }
    }
    
    imshow("Pseudo Colored", im_out);
    waitKey(0);
}