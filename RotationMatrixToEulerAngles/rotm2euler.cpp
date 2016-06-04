/*
 * Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
 * All rights reserved. No warranty, explicit or implicit, provided.
 */

#include "opencv2/opencv.hpp"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace cv;
using namespace std;

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
    
    return  norm(I, shouldBeIdentity) < 1e-6;
    
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat &R)
{

    assert(isRotationMatrix(R));
    
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
    
    
    
}

// Calculates rotation matrix given euler angles.
Mat eulerAnglesToRotationMatrix(Vec3f &theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
    
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
    
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
    
    
    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;
    
    return R;

}



int main(int argc, char** argv)
{
    // Initialize random number generator
    srand (time(NULL));
    
    // Randomly generate Euler angles in Degrees.
    Vec3f eDegrees(rand() % 360 - 180.0, rand() % 360 - 180.0, rand() % 360 - 180.0);

    // Convert angles to radians
    Vec3f e = eDegrees * M_PI / 180.0;

    // Calculate rotation matrix
    Mat R = eulerAnglesToRotationMatrix(e);
    
    // Calculate Euler angles from rotation matrix
    Vec3f e1 = rotationMatrixToEulerAngles(R);
    
    // Calculate rotation matrix
    Mat R1 = eulerAnglesToRotationMatrix(e1);

    // Note e and e1 will be the same a lot of times
    // but not always. R and R1 should be the same always.
    
    cout << endl << "Input Angles" << endl << e << endl;
    cout << endl << "R : " << endl << R << endl;
    cout << endl << "Output Angles" << endl << e1 << endl;
    cout << endl << "R1 : " << endl << R1 << endl;
    

    
}