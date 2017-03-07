/**
 Copyright 2017 by Satya Mallick ( Big Vision LLC )
 http://www.learnopencv.com
**/

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void fillHoles(Mat &mask)
{
    /* 
     This hole filling algorithm is decribed in this post
     https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
     */
     
    Mat maskFloodfill = mask.clone();
    floodFill(maskFloodfill, cv::Point(0,0), Scalar(255));
    Mat mask2;
    bitwise_not(maskFloodfill, mask2);
    mask = (mask2 | mask);

}

int main(int argc, char** argv )
{
    // Read image
    Mat img = imread("red_eyes2.jpg",CV_LOAD_IMAGE_COLOR);

    // Output image
    Mat imgOut = img.clone();
    
    // Load HAAR cascade
    CascadeClassifier eyesCascade("haarcascade_eye.xml");
    
    // Detect eyes
    std::vector<Rect> eyes;
    eyesCascade.detectMultiScale( img, eyes, 1.3, 4, 0 |CASCADE_SCALE_IMAGE, Size(100, 100) );

    
    // For every detected eye
    for( size_t i = 0; i < eyes.size(); i++ )
    {
    
        // Extract eye from the image.
        Mat eye = img(eyes[i]);
        
        // Split eye image into 3 channels.
        vector<Mat>bgr(3);
        split(eye,bgr);
        
        // Simple red eye detector
        Mat mask = (bgr[2] > 150) & (bgr[2] > ( bgr[1] + bgr[0] ));

        // Clean mask -- 1) File holes 2) Dilate (expand) mask
        fillHoles(mask);
        dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);

        

        // Calculate the mean channel by averaging
        // the green and blue channels

        Mat mean = (bgr[0]+bgr[1])/2;
        mean.copyTo(bgr[2], mask);
        mean.copyTo(bgr[0], mask);
        mean.copyTo(bgr[1], mask);

        // Merge channels
        Mat eyeOut;
        cv::merge(bgr,eyeOut);

        // Copy the fixed eye to the output image.
        eyeOut.copyTo(imgOut(eyes[i]));

    }

    // Display Result
    imshow("Red Eyes", img);
    imshow("Red Eyes Removed", imgOut);
    waitKey(0);

}
