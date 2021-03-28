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
int dispType = CV_16S;


// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();

cv::Mat imgL;
cv::Mat imgR;
cv::Mat imgL_gray;
cv::Mat imgR_gray;


// Defining callback functions for the trackbars to update parameter values

static void on_trackbar1( int, void* )
{
  stereo->setNumDisparities(numDisparities*16);
  numDisparities = numDisparities*16;
}

static void on_trackbar2( int, void* )
{
  stereo->setBlockSize(blockSize*2+5);
  blockSize = blockSize*2+5;
}

static void on_trackbar3( int, void* )
{
  stereo->setPreFilterType(preFilterType);
}

static void on_trackbar4( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;
}

static void on_trackbar5( int, void* )
{
  stereo->setPreFilterCap(preFilterCap);
}

static void on_trackbar6( int, void* )
{
  stereo->setTextureThreshold(textureThreshold);
}

static void on_trackbar7( int, void* )
{
  stereo->setUniquenessRatio(uniquenessRatio);
}

static void on_trackbar8( int, void* )
{
  stereo->setSpeckleRange(speckleRange);
}

static void on_trackbar9( int, void* )
{
  stereo->setSpeckleWindowSize(speckleWindowSize*2);
  speckleWindowSize = speckleWindowSize*2;
}

static void on_trackbar10( int, void* )
{
  stereo->setDisp12MaxDiff(disp12MaxDiff);
}

static void on_trackbar11( int, void* )
{
  stereo->setMinDisparity(minDisparity);
}


int main()
{
  // Initialize variables to store the maps for stereo rectification
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
  if (!camL.isOpened())
  {
    std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    return -1;
  }

  // Creating a named window that can be linked to the trackbars
  cv::namedWindow("disparity",cv::WINDOW_NORMAL);
  cv::resizeWindow("disparity",600,600);

  // Creating trackbars to dynamically update the StereoBM parameters
  cv::createTrackbar("numDisparities", "disparity", &numDisparities, 18, on_trackbar1);
  cv::createTrackbar("blockSize", "disparity", &blockSize, 50, on_trackbar2);
  cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
  cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
  cv::createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
  cv::createTrackbar("textureThreshold", "disparity", &textureThreshold, 100, on_trackbar6);
  cv::createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar7);
  cv::createTrackbar("speckleRange", "disparity", &speckleRange, 100, on_trackbar8);
  cv::createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 25, on_trackbar9);
  cv::createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar10);
  cv::createTrackbar("minDisparity", "disparity", &minDisparity, 25, on_trackbar11);

  cv::Mat disp, disparity;

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
    // is essential to convert it to CV_32F and scale it down 16 times.

    // Converting disparity values to CV_32F from CV_16S
    disp.convertTo(disparity,CV_32F, 1.0);

    // Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

    // Displaying the disparity map
    cv::imshow("disparity",disparity);

    // Close window using esc key
    if (cv::waitKey(1) == 27) break;
  }
  
  std::cout << "Saving depth estimation paraeters ......";

  cv::FileStorage cv_file = cv::FileStorage("../data/depth_estimation_params_cpp.xml", cv::FileStorage::WRITE);
  cv_file.write("numDisparities",numDisparities);
  cv_file.write("blockSize",blockSize);
  cv_file.write("preFilterType",preFilterType);
  cv_file.write("preFilterSize",preFilterSize);
  cv_file.write("preFilterCap",preFilterCap);
  cv_file.write("textureThreshold",textureThreshold);
  cv_file.write("uniquenessRatio",uniquenessRatio);
  cv_file.write("speckleRange",speckleRange);
  cv_file.write("speckleWindowSize",speckleWindowSize);
  cv_file.write("disp12MaxDiff",disp12MaxDiff);
  cv_file.write("minDisparity",minDisparity);
  cv_file.write("M",39.075);
  cv_file.release();

  return 0;
}
