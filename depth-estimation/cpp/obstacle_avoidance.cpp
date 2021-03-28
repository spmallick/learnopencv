#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <iomanip>

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
float M = 0.0;


cv::Mat imgL;
cv::Mat imgR;
cv::Mat imgL_gray;
cv::Mat imgR_gray;
cv::Mat disp, disparity, depth_map;
cv::Mat output_canvas;

// These parameters can vary according to the setup
float max_depth = 400.0; //maximum distance the setup can measure (in cm)
float min_depth = 50.0; //minimum distance the setup can measure (in cm)
float depth_thresh = 100.0; // Threshold for SAFE distance (in cm)

// function to sort contours from largest to smallest
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( cv::contourArea(cv::Mat(contour1)) );
    double j = fabs( cv::contourArea(cv::Mat(contour2)) );
    return ( i > j );
}

void obstacle_avoid()
{
  cv::Mat mask, mean, stddev, mask2;

  // Mask to segment regions with depth less than safe distance
  cv::inRange(depth_map, 10, depth_thresh, mask);
  double s = (cv::sum(mask)[0])/255.0;
  double img_area = double(mask.rows * mask.cols);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  // Check if a significantly large obstacle is present and filter out smaller noisy regions
  if (s > 0.01*img_area)
  {
    // finding conoturs in the generated mask
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    // sorting contours from largest to smallest
    std::sort(contours.begin(), contours.end(), compareContourAreas);

    // extracting the largest contour
    std::vector<cv::Point> cnt = contours[0];

    // Check if detected contour is significantly large (to avoid multiple tiny regions)
    double cnt_area = fabs( cv::contourArea(cv::Mat(cnt)));
    if (cnt_area > 0.01*img_area)
    {
      cv::Rect box;

      // Finding the bounding rectangle for the largest contour
      box = cv::boundingRect(cnt);

      // finding average depth of region represented by the largest contour
      mask2 = mask*0;
      cv::drawContours(mask2, contours, 0, (255), -1);

      // Calculating the average depth of the object closer than the safe distance
      cv::meanStdDev(depth_map, mean, stddev, mask2);

      // Printing the warning text with object distance
      char text[10];
      std::sprintf(text, "%.2f cm",mean.at<double>(0,0));

      cv::putText(output_canvas, "WARNING!", cv::Point2f(box.x + 5, box.y-40), 1, 2, cv::Scalar(0,0,255), 2, 2);
      cv::putText(output_canvas, "Object at", cv::Point2f(box.x + 5, box.y), 1, 2, cv::Scalar(0,0,255), 2, 2);
      cv::putText(output_canvas, text, cv::Point2f(box.x + 5, box.y+40), 1, 2, cv::Scalar(0,0,255), 2, 2);

    }
  }
  else
  {
    // Printing SAFE if no obstacle is closer than the safe distance
    cv::putText(output_canvas, "SAFE!", cv::Point2f(200,200),1,2,cv::Scalar(0,255,0),2,2);
  }


  // Displaying the output of the obstacle avoidance system
  cv::imshow("output_canvas",output_canvas);
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
  cv_file["M"] >> M;
  
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
  if (!camL.isOpened())
  {
    std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    return -1;
  }

  cv::namedWindow("disparity",cv::WINDOW_NORMAL);
  cv::resizeWindow("disparity",600,600);

  while (true)
  {
    // Capturing and storing left and right camera images
    camL >> imgL;
    camR >> imgR;

    // copy left image to display text message for the obstacle avoidance system
    imgL.copyTo(output_canvas);

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

    // Calculating disparity to depth map using the following equation
    // ||    depth = M * (1/disparity)   ||
    depth_map = (float)M/disparity;

    // Updating the output of the obstacle avoidance system
    obstacle_avoid();

    // Displaying the disparity map
    cv::imshow("disparity",disparity);

    // Close window using esc key
    if (cv::waitKey(1) == 27) break;

  }

  return 0;
}
