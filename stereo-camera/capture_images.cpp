#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <iostream>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

int main()
{
  std::cout << "Checking the right and left camera IDs:" << std::endl;
  std::cout << "Press (y) if IDs are correct and (n) to swap the IDs" << std::endl;
  std::cin.get();

  // Check for left and right camera IDs
  int CamL_id{0}, CamR_id{1}; // These values can change depending on the system

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

  cv::Mat frameL, frameR, temp;
  cv::namedWindow("left and right frames", cv::WINDOW_NORMAL);
  cv::resizeWindow("left and right frames", 1200, 600);

  for (size_t i{0}; i<100; i++)
  {
    camL >> frameL;
    camR >> frameR;
  }

  cv::hconcat(frameL, frameR, temp);
  cv::imshow("left and right frames", temp);
  char c = (char)cv::waitKey(0);

  // temp.release();

  if (c == 'y')
    std::cout << "camera IDs retained" << std::endl;
  else if (c == 'n')
  {
    int temp;
    temp = CamL_id;
    CamL_id = CamR_id;
    CamR_id = temp;
    std::cout << "camera IDs swapped" << std::endl;
  }

  else
  {
    std::cout << "Wrong response!!!" << std::endl;
    return -1;
  }

  camL.release();
  camR.release();

  std::string output_path{"./data_cpp/"};

  cv::VideoCapture camL_valid(CamL_id), camR_valid(CamR_id);

  int64 start{cv::getTickCount()};
  float time{0};

  cv::Mat grayL, grayR;

  std::vector<cv::Point2f> cornersL, cornersR;
  cv::Size board = cv::Size(9, 6);
  bool foundL, foundR;
  int count{0};

  for (size_t i{0}; i<100000; i++)
  {
    camL_valid >> frameL;
    camR_valid >> frameR;

    time = (cv::getTickCount() - start)/cv::getTickFrequency();

    cv::hconcat(frameL, frameR, temp);
    cv::putText(temp, std::to_string(time), cv::Point(50, 100), 2, 1, cv::Scalar(0, 200, 200), 2);
    cv::imshow("left and right frames", temp);
    cv::waitKey(1);

    cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);

    foundL = cv::findChessboardCorners(grayL, board, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
    foundR = cv::findChessboardCorners(grayR, board, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

    std::cout << foundL << "," << foundR << std::endl;

    if ((foundL==true) && (foundR==true) && (time>10))
    {
      count++;
      cv::imwrite(output_path+"stereoL/img_"+std::to_string(count)+".jpg", frameL);
      cv::imwrite(output_path+"stereoR/img_"+std::to_string(count)+".jpg", frameL);
      std::cout << output_path+"stereoL/img_"+std::to_string(count)+".jpg" << std::endl;
    }

    if (time > 10)
      start = cv::getTickCount();
  }

  return 0;
}