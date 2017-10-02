//
//  alphaBlend.cpp
//  
//
//  Created by Sunita Nayak on 3/14/17.
//
//

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Alpha blending using multiply and add functions
Mat& blend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
    Mat fore, back;
    multiply(alpha, foreground, fore);
    multiply(Scalar::all(1.0)-alpha, background, back);
    add(fore, back, outImage);
    
    return outImage;
}

// Alpha Blending using direct pointer access
Mat& alphaBlendDirectAccess(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{

    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();
    
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    int i,j;
    for ( j = 0; j < numberOfPixels; ++j, outImagePtr++, fptr++, aptr++, bptr++)
    {
        *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
    }
    
    return outImage;
}


int main(int argc, char** argv)
{
    
    // Read in the png foreground asset file that contains both rgb and alpha information
    Mat foreGroundImage = imread("foreGroundAssetLarge.png", -1);
    Mat bgra[4];
    split(foreGroundImage, bgra);//split png foreground
    
    // Save the foregroung RGB content into a single Mat
    vector<Mat> foregroundChannels;
    foregroundChannels.push_back(bgra[0]);
    foregroundChannels.push_back(bgra[1]);
    foregroundChannels.push_back(bgra[2]);
    Mat foreground = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(foregroundChannels, foreground);
    
    // Save the alpha information into a single Mat
    vector<Mat> alphaChannels;
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    Mat alpha = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(alphaChannels, alpha);
   
    // Read background image
    Mat background = imread("backGroundLarge.jpg");
    
    // Convert Mat to float data type
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0/255); // keeps the alpha values betwen 0 and 1

    // Number of iterations to average the performane over
    int numOfIterations = 1; //1000;
    
    // Alpha blending using functions multiply and add
    Mat outImage= Mat::zeros(foreground.size(), foreground.type());
    double t = (double)getTickCount();
    for (int i=0; i<numOfIterations; i++) {
        outImage = blend(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Time for alpha blending using multiply & add function : " << t*1000/numOfIterations << " milliseconds" << endl;

    // Alpha blending using direct Mat access with for loop
    outImage = Mat::zeros(foreground.size(), foreground.type());
    t = (double)getTickCount();
    for (int i=0; i<numOfIterations; i++) {
        outImage = alphaBlendDirectAccess(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Time for alpha blending using alphaBlendDirectAccess : " << t*1000/numOfIterations << " milliseconds" << endl;

    //imshow("alpha blended image", outImage/255);
    //waitKey(0);
    
    return 0;
}
