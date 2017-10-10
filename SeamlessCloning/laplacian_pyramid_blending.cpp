/**
 * OpenCV Laplacian Pyramid Blending Example
 *
 * Copyright 2017 by Satya Mallick <spmallick@gmail.com>
 *
 */

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void getLaplacianPyramid(Mat& gaussianPyramid, Mat& laplacianPyramid){
    // compute laplacian pyramid from the gaussian pyramid
    Mat downSampled;
    pyrDown(gaussianPyramid,downSampled);

    // up sample the down sampled
    Mat blurred;
    pyrUp(downSampled,blurred);

    subtract(gaussianPyramid, blurred, laplacianPyramid);

}

void combineImages(Mat& A, Mat& B, Mat& destination){
    // Get left half of A
    int widthA = A.size().width;
    int heightA = A.size().height;
    Mat halfA = A(Rect(0,0,widthA/2,heightA));

    // Get right half of B
    int widthB = B.size().width;
    int heightB = B.size().height;
    Mat halfB = B(Rect(widthA/2,0,widthB-widthA/2,heightB));

    // Concatenate both halves
    hconcat(halfA, halfB, destination);

}

int main( int argc, char** argv )
{
    // Read two images
    Mat A = imread("images/apple.png");
    Mat B = imread("images/orange.jpg");
    
    // Start with original images (base of pyramids)
    Mat gaussianA = A.clone();
    Mat gaussianB = B.clone();

    // Number of levels in pyramids, try with different values 
    int maxIterations = 6;

    // Combined laplacian pyramids of both images
    vector<Mat> combinedLaplacianPyramids;
    
    for (int i = 0; i < maxIterations; i++){

        // compute laplacian pyramids for A
        Mat laplacianA;
        getLaplacianPyramid(gaussianA,laplacianA);

        // compute laplacian pyramids for B
        Mat laplacianB;
        getLaplacianPyramid(gaussianB,laplacianB);

        // combine laplacian pyramids
        Mat combinedLaplacian;
        combineImages(laplacianA, laplacianB, combinedLaplacian);

        // Insert combinedLaplacian in the beginning of the list of combined laplacian pyramids
        combinedLaplacianPyramids.insert(combinedLaplacianPyramids.begin(),combinedLaplacian);

        // Update guassian pyramids for next iteration
        pyrDown(gaussianA,gaussianA);
        pyrDown(gaussianB,gaussianB);

    }
 
    // combine last gaussians (top level of laplacian pyramids is same as that of gaussian's)
    Mat lastCombined;
    combineImages(gaussianA, gaussianB, lastCombined);

    // Insert lastCombined in the beginning of the list of combined laplacian pyramids
    combinedLaplacianPyramids.insert(combinedLaplacianPyramids.begin(),lastCombined);

    // reconstructing image
    Mat blendedImage = combinedLaplacianPyramids[0];

    for (int i = 1; i < combinedLaplacianPyramids.size(); i++){
        // upSample and add to next level
        pyrUp(blendedImage,blendedImage);
        add(blendedImage, combinedLaplacianPyramids[i],blendedImage);
    }

    // direct combining both halves for comparison
    Mat directCombined; 
    combineImages(A, B, directCombined);

    // view both images and compare
    imshow("Blended",blendedImage);
    imshow("Direct combination",directCombined);
    waitKey(0);

}
