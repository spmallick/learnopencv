/**
 * OpenCV SimpleBlobDetector Example
 *
 * Copyright 2015 by Satya Mallick <spmallick@gmail.com>
 *
 */

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

string type2str(int type) {
    string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    return r;
}


int main( int argc, char** argv )
{
    // Read 8-bit grayscale image
    Mat im = imread("earth-16-bit-per-channel.png", IMREAD_GRAYSCALE);
    cout << "flags : IMREAD_GRAYSCALE" << endl;
    cout << "Number of Channels : " << im.channels() << ", depth : " << type2str(im.depth()) << endl << endl;
    
    // Read 8-bit color image
    im =  imread("earth-16-bit-per-channel.png", IMREAD_COLOR);
    cout <<  "flags : IMREAD_COLOR" << endl;
    cout << "Number of Channels : " << im.channels() << ", depth : " << type2str(im.depth()) << endl << endl;

    // Read 16-bit color image
    im = imread("earth-16-bit-per-channel.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH );
    cout <<  "flags :  IMREAD_ANYCOLOR | IMREAD_ANYDEPTH" << endl;
    cout << "Number of Channels : " << im.channels() << ", depth : " << type2str(im.depth()) << endl << endl;

    // Read transparent PNG / TIFF image
    im = imread("earth-16-bit-per-channel.png", IMREAD_UNCHANGED);
    cout << "flags : IMREAD_UNCHANGED" << endl;
    cout << "Number of Channels : " << im.channels() << ", depth : " << type2str(im.depth()) << endl << endl;
}
