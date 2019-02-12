#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main(){
 
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap(0); 
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  Mat background;
  for(int i=0;i<60;i++)
  {
    cap >> background;
  }
  
  flip(background,background,1);

  while(1)
  {
 
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
    
    Mat hsv;
    flip(frame,frame,1);
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Mat mask1,mask2;
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);

    mask1 = mask1 + mask2;
	
	  Mat kernel = Mat::ones(3,3, CV_32F);
    morphologyEx(mask1,mask1,cv::MORPH_OPEN,kernel);
    morphologyEx(mask1,mask1,cv::MORPH_DILATE,kernel);
    
	  bitwise_not(mask1,mask2);
    
	  Mat res1, res2, final_output;
    bitwise_and(frame,frame,res1,mask2);
    bitwise_and(background,background,res2,mask1);
	  addWeighted(res1,1,res2,1,0,final_output);
	


    imshow("Magic !!!",final_output);
    // Display the resulting frame
    //imshow( "Frame", frame );

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
	  // Also relese all the mat created in the code to avoid memory leakage.
  	frame.release(),hsv.release(),mask1.release(),mask2.release(),res1.release(),res2.release(),final_output.release();
 
  }
  
  // When everything done, release the video capture object
  cap.release();
  
  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}