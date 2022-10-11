#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"

// initialize values for StereoSGBM parameters
int numDisparities = 8;
int blockSize = 5;
int preFilterType = 1;
int preFilterSize = 1;
int preFilterCap = 31;
int minDisparity = 0;
int textureThreshold = 10;
int uniquenessRatio = 15;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = -1;


cv::Mat imgL;
cv::Mat imgR;
cv::Mat imgL_gray;
cv::Mat imgR_gray;
cv::Mat disp, disparity, disp_8;

std::vector<float> z_vec;
std::vector<cv::Point2f> coeff_vec;

// These parameters can vary according to the setup
// Keeping the target object at max_dist we store disparity values
// after every sample_delta distance.
int max_dist = 230; // max distance to keep the target object (in cm)
int min_dist = 50; // Minimum distance the stereo setup can measure (in cm)
int sample_delta = 40; // Distance between two sampling points (in cm)
float Z = max_dist;

// Defining callback functions for mouse events
void mouseEvent(int evt, int x, int y, int flags, void* param) {                    
    float depth_val;

    if (evt == CV_EVENT_LBUTTONDOWN) {
      depth_val  = disparity.at<float>(y,x);

      if (depth_val > 0)
      {
        z_vec.push_back(Z);
        coeff_vec.push_back(cv::Point2f(1.0f/(float)depth_val, 1.0f));
        Z = Z-sample_delta;
      }
    }         
}

int main()
{

  // Creating an object of StereoBM algorithm
  cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();

  // Reading the stored the StereoBM parameters
  cv::FileStorage cv_file = cv::FileStorage("../data/depth_estimation_params_cpp.xml", cv::FileStorage::READ);
  cv_file["numDisparities"] >> numDisparities;
  cv_file["blockSize"] >> blockSize;
  cv_file["preFilterType"] >> preFilterType;
  cv_file["preFilterSize"] >> preFilterSize;
  cv_file["preFilterCap"] >> preFilterCap;
  cv_file["minDisparity"] >> minDisparity;
  cv_file["textureThreshold"] >> textureThreshold;
  cv_file["uniquenessRatio"] >> uniquenessRatio;
  cv_file["speckleRange"] >> speckleRange;
  cv_file["speckleWindowSize"] >> speckleWindowSize;
  cv_file["disp12MaxDiff"] >> disp12MaxDiff;
  
  // updating the parameter values of the StereoBM algorithm
  stereo->setNumDisparities(numDisparities);
	stereo->setBlockSize(blockSize);
	stereo->setPreFilterType(preFilterType);
	stereo->setPreFilterSize(preFilterSize);
	stereo->setPreFilterCap(preFilterCap);
	stereo->setTextureThreshold(textureThreshold);
	stereo->setUniquenessRatio(uniquenessRatio);
	stereo->setSpeckleRange(speckleRange);
	stereo->setSpeckleWindowSize(speckleWindowSize);
	stereo->setDisp12MaxDiff(disp12MaxDiff);
	stereo->setMinDisparity(minDisparity);

  //Initialize variables to store the maps for stereo rectification
  cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
  cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

  // Reading the mapping values for stereo image rectification
  cv::FileStorage cv_file2 = cv::FileStorage("../data/stereo_rectify_maps.xml", cv::FileStorage::READ);
  cv_file2["Left_Stereo_Map_x"] >> Left_Stereo_Map1;
  cv_file2["Left_Stereo_Map_y"] >> Left_Stereo_Map2;
  cv_file2["Right_Stereo_Map_x"] >> Right_Stereo_Map1;
  cv_file2["Right_Stereo_Map_y"] >> Right_Stereo_Map2;
  cv_file2.release();

  // Check for left and right camera IDs
  // These values can change depending on the system
  int CamL_id{2}; // Camera ID for left camera
  int CamR_id{0}; // Camera ID for right camera

  cv::VideoCapture camL(CamL_id), camR(CamR_id);

  // Check if left camera is attched
  if (!camL.isOpened())
  {
    std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    return -1;
  }

  // Check if right camera is attached
  if (!camR.isOpened())
  {
    std::cout << "Could not open camera with index : " << CamR_id << std::endl;
    return -1;
  }

  cv::namedWindow("disparity",cv::WINDOW_NORMAL);
  cv::resizeWindow("disparity",600,600);
  cv::setMouseCallback("disparity", mouseEvent, NULL);

  while (true)
  {
    // Capturing and storing left and right camera images
    camL >> imgL;
    camR >> imgR;

    // Converting images to grayscale
    cv::cvtColor(imgL, imgL_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, imgR_gray, cv::COLOR_BGR2GRAY);

    // Initialize matrix for rectified stero images
    cv::Mat Left_nice, Right_nice;

    // Applying stereo image rectification on the left image
    cv::remap(imgL_gray,
              Left_nice,
              Left_Stereo_Map1,
              Left_Stereo_Map2,
              cv::INTER_LANCZOS4,
              cv::BORDER_CONSTANT,
              0);
    // Applying stereo image rectification on the right image
    cv::remap(imgR_gray,
              Right_nice,
              Right_Stereo_Map1,
              Right_Stereo_Map2,
              cv::INTER_LANCZOS4,
              cv::BORDER_CONSTANT,
              0);

    // Calculating disparith using the StereoBM algorithm
    stereo->compute(Left_nice,Right_nice,disp);

    // NOTE: compute returns a 16bit signed single channel image,
		// CV_16S containing a disparity map scaled by 16. Hence it 
    // is essential to convert it to CV_16S and scale it down 16 times.

    // Converting disparity values to CV_32F from CV_16S
    disp.convertTo(disparity,CV_32F, 1.0);

    // Scaling down the disparity values and normalizing them
    disparity = (disparity/(float)16.0 - (float)minDisparity)/((float)numDisparities);

    // Displaying the disparity map
    cv::imshow("disparity",disparity);

    // Close window using esc key
    if (cv::waitKey(1) == 27) break;

    // Close program after taking reading for the last data point
    if (Z <= min_dist) break;

  }

  //solving for M in the following equation
  //||    depth = M * (1/disparity)   ||
  //for N data points coeff is Nx2 matrix with values 
  //1/disparity, 1
  //and depth is Nx1 matrix with depth values

  cv::Mat Z_mat(z_vec.size(), 1, CV_32F, z_vec.data());
  cv::Mat coeff(z_vec.size(), 2, CV_32F, coeff_vec.data());

  cv::Mat sol(2, 1, CV_32F);
  float M;

  // Solving for M using least square fitting with QR decomposition method 
  cv::solve(coeff, Z_mat, sol, cv::DECOMP_QR);

  M = sol.at<float>(0,0);

  // Storing the updated value of M along with the stereo parameters
  cv::FileStorage cv_file3 = cv::FileStorage("../data/depth_estimation_params_cpp.xml", cv::FileStorage::WRITE);
  cv_file3.write("numDisparities",numDisparities);
  cv_file3.write("blockSize",blockSize);
  cv_file3.write("preFilterType",preFilterType);
  cv_file3.write("preFilterSize",preFilterSize);
  cv_file3.write("preFilterCap",preFilterCap);
  cv_file3.write("textureThreshold",textureThreshold);
  cv_file3.write("uniquenessRatio",uniquenessRatio);
  cv_file3.write("speckleRange",speckleRange);
  cv_file3.write("speckleWindowSize",speckleWindowSize);
  cv_file3.write("disp12MaxDiff",disp12MaxDiff);
  cv_file3.write("minDisparity",minDisparity);
  cv_file3.write("M",M);
  cv_file3.release();
  
  return 0;
}
