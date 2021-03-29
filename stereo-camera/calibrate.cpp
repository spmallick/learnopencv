#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

int main()
{
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpointsL, imgpointsR;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j,i,0));
  }

  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> imagesL, imagesR;
  // Path of the folder containing checkerboard images
  std::string pathL = "./data/stereoL/*.png";
  std::string pathR = "./data/stereoR/*.png";

  cv::glob(pathL, imagesL);
  cv::glob(pathR, imagesR);

  cv::Mat frameL, frameR, grayL, grayR;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_ptsL, corner_ptsR;
  bool successL, successR;

  // Looping over all the images in the directory
  for(int i{0}; i<imagesL.size(); i++)
  {
    frameL = cv::imread(imagesL[i]);
    cv::cvtColor(frameL,grayL,cv::COLOR_BGR2GRAY);

    frameR = cv::imread(imagesR[i]);
    cv::cvtColor(frameR,grayR,cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    successL = cv::findChessboardCorners(
      grayL,
      cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]),
      corner_ptsL);
      // cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

    successR = cv::findChessboardCorners(
      grayR,
      cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]),
      corner_ptsR);
      // cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
    /*
      * If desired number of corner are detected,
      * we refine the pixel coordinates and display 
      * them on the images of checker board
    */
    if((successL) && (successR))
    {
      cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(grayL,corner_ptsL,cv::Size(11,11), cv::Size(-1,-1),criteria);
      cv::cornerSubPix(grayR,corner_ptsR,cv::Size(11,11), cv::Size(-1,-1),criteria);

      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frameL, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL,successL);
      cv::drawChessboardCorners(frameR, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR,successR);

      objpoints.push_back(objp);
      imgpointsL.push_back(corner_ptsL);
      imgpointsR.push_back(corner_ptsR);
    }

    cv::imshow("ImageL",frameL);
    cv::imshow("ImageR",frameR);
    cv::waitKey(0);
  }

  cv::destroyAllWindows();

  cv::Mat mtxL,distL,R_L,T_L;
  cv::Mat mtxR,distR,R_R,T_R;
  

  /*
    * Performing camera calibration by 
    * passing the value of known 3D points (objpoints)
    * and corresponding pixel coordinates of the 
    * detected corners (imgpoints)
  */

  cv::Mat new_mtxL, new_mtxR;

  // Calibrating left camera
  cv::calibrateCamera(objpoints,
                      imgpointsL,
                      grayL.size(),
                      mtxL,
                      distL,
                      R_L,
                      T_L);

  new_mtxL = cv::getOptimalNewCameraMatrix(mtxL,
                                distL,
                                grayL.size(),
                                1,
                                grayL.size(),
                                0);

  // Calibrating right camera
  cv::calibrateCamera(objpoints,
                      imgpointsR,
                      grayR.size(),
                      mtxR,
                      distR,
                      R_R,
                      T_R);

  new_mtxR = cv::getOptimalNewCameraMatrix(mtxR,
                                distR,
                                grayR.size(),
                                1,
                                grayR.size(),
                                0);
  
  // Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat 
  // are calculated. Hence intrinsic parameters are the same.
  cv::Mat Rot, Trns, Emat, Fmat;

  int flag = 0;
  flag |= cv::CALIB_FIX_INTRINSIC;


  // This step is performed to transformation between the two cameras and calculate Essential and 
  // Fundamenatl matrix
  cv::stereoCalibrate(objpoints,
                      imgpointsL,
                      imgpointsR,
                      new_mtxL,
                      distL,
                      new_mtxR,
                      distR,
                      grayR.size(),
                      Rot,
                      Trns,
                      Emat,
                      Fmat,
                      flag,
                      cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 1e-6));

  cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;

  // Once we know the transformation between the two cameras we can perform 
  // stereo rectification
  cv::stereoRectify(new_mtxL,
                    distL,
                    new_mtxR,
                    distR,
                    grayR.size(),
                    Rot,
                    Trns,
                    rect_l,
                    rect_r,
                    proj_mat_l,
                    proj_mat_r,
                    Q,
                    1);

  // Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
  // Compute the rectification map (mapping between the original image pixels and 
  // their transformed values after applying rectification and undistortion) for left and right camera frames
  cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
  cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

  cv::initUndistortRectifyMap(new_mtxL,
                              distL,
                              rect_l,
                              proj_mat_l,
                              grayR.size(),
                              CV_16SC2,
                              Left_Stereo_Map1,
                              Left_Stereo_Map2);

  cv::initUndistortRectifyMap(new_mtxR,
                              distR,
                              rect_r,
                              proj_mat_r,
                              grayR.size(),
                              CV_16SC2,
                              Right_Stereo_Map1,
                              Right_Stereo_Map2);

  cv::FileStorage cv_file = cv::FileStorage("data/params_cpp.xml", cv::FileStorage::WRITE);
  cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map1);
  cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map2);
  cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map1);
  cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map2);
  cv_file.release();
  return 0;
}