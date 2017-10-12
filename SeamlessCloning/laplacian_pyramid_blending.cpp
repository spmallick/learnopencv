/**
 * OpenCV Laplacian Pyramid Blending Example
 *
 * Copyright 2017 by Satya Mallick <spmallick@gmail.com>
 *
 */

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void getLaplacianPyramid(Mat& guassianPyramid, Mat& laplacianPyramid){
    // compute laplacian pyramid from the guassian pyramid
    Mat downSampled;
    pyrDown(guassianPyramid,downSampled);

    // up sample the down sampled
    Mat blurred;
    pyrUp(downSampled,blurred);

    subtract(guassianPyramid, blurred, laplacianPyramid);

}

void combineImages(Mat& A, Mat& B, Mat& mask, Mat& destination){
    
    destination = Mat::zeros(A.rows, A.cols, CV_32FC3);
    
    // destination is weighted sum of A and B, with weights mask, and 1-mask respectively
    for(int y = 0; y < A.rows; y++)
    {
        for(int x = 0; x < A.cols; x++)
        {   
            Vec3f a = A.at<Vec3f>(Point(x,y));
            Vec3f b = B.at<Vec3f>(Point(x,y));
            Vec3f m = mask.at<Vec3f>(Point(x,y));
            
            float b_ = a[0]*m[0]+(1-m[0])*b[0];
            float g_ = a[1]*m[1]+(1-m[1])*b[1];
            float r_ = a[2]*m[2]+(1-m[2])*b[2];

            destination.at<Vec3f>(y,x)[0] = b_;
            destination.at<Vec3f>(y,x)[1] = g_;
            destination.at<Vec3f>(y,x)[2] = r_;
        }
    }
}

int main( int argc, char** argv )
{
    // Read two images
    Mat A = imread("images/man.jpg");
    Mat B = imread("images/woman.jpg");

    // Convert to float
    A.convertTo(A, CV_32FC3, 1/255.0);
    B.convertTo(B, CV_32FC3, 1/255.0);

    // Create a rough mask around man's face in A.
    Mat mask = Mat::zeros(A.rows, A.cols, CV_8UC3);

    // Create some points around airplane
    Point points[11];
    points[0] = Point(164,226);
    points[1] = Point(209,225);
    points[2] = Point(238,188);
    points[3] = Point(252,133);
    points[4] = Point(248,75);
    points[5] = Point(240,29);
    points[6] = Point(192,15);
    points[7] = Point(150,15);
    points[8] = Point(100,70);
    points[9] = Point(106,133);
    points[10] = Point(123,194);

    const Point* polygon[1] = {points}; //Array of points arrays
    int npt[] = {11}; // Length of points array
    
    //Fill the polygon formed by the points 
    fillPoly(mask, polygon, npt, 1, Scalar(255, 255, 255));

    // Convert to float
    mask.convertTo(mask, CV_32FC3, 1/255.0);

    // Multiply with float < 1.0 to take weighted average of man and woman's face
    mask = mask * 0.7;

    // Resizing to multiples of 2^(levels in pyramid), thus 32 in our case
    resize(A, A, Size(384,352));

    // B and mask should have same size as A for multiplication and addition operations later
    resize(B, B, A.size());
    resize(mask, mask, A.size());

    // Start with original images (base of pyramids)
    Mat guassianA = A.clone();
    Mat guassianB = B.clone();
    Mat guassianMask = mask.clone();

    // Number of levels in pyramids, try with different values. Be careful with image sizes
    int maxIterations = 2;

    // Combined laplacian pyramids of both images
    vector<Mat> combinedLaplacianPyramids;

    for (int i = 0; i < maxIterations; i++){
        // compute laplacian pyramids for A
        Mat laplacianA;
        getLaplacianPyramid(guassianA,laplacianA);

        // compute laplacian pyramids for B
        Mat laplacianB;
        getLaplacianPyramid(guassianB,laplacianB);

        // combine laplacian pyramids
        Mat combinedLaplacian;
        combineImages(laplacianA, laplacianB, guassianMask, combinedLaplacian);
 
        // Insert combinedLaplacian in the beginning of the list of combined laplacian pyramids
        combinedLaplacianPyramids.insert(combinedLaplacianPyramids.begin(),combinedLaplacian);

        // Update guassian pyramids for next iteration
        pyrDown(guassianA,guassianA);
        pyrDown(guassianB,guassianB);
        pyrDown(guassianMask,guassianMask);

    }

    // combine last guassians (top level of laplacian pyramids is same as that of guassian's)
    Mat lastCombined;
    combineImages(guassianA, guassianB, guassianMask, lastCombined);

    // Insert lastCombined in the beginning of the list of combined laplacian pyramids
    combinedLaplacianPyramids.insert(combinedLaplacianPyramids.begin(),lastCombined);

    // reconstructing image
    Mat blendedImage = combinedLaplacianPyramids[0];

    for (int i = 1; i < combinedLaplacianPyramids.size(); i++){
        // upSample and add to next level
        pyrUp(blendedImage,blendedImage);
        add(blendedImage, combinedLaplacianPyramids[i],blendedImage);
    }

    // put blended image back to sky image at original location
    imshow("blended",blendedImage);

    // direct combining both halves for comparison
    Mat directCombined;
    combineImages(A, B, mask, directCombined);
    imshow("directCombined",directCombined);
    waitKey(0);


    return 0;
}
