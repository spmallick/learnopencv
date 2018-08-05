/*
  Copyright 2018 BIG VISION LLC ALL RIGHTS RESERVED
*/

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

using namespace cv;
using namespace std;

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"}; 

// create tracker by name
Ptr<Tracker> createTrackerByName(string trackerType) 
{
  Ptr<Tracker> tracker;
  if (trackerType ==  trackerTypes[0])
    tracker = TrackerBoosting::create();
  else if (trackerType == trackerTypes[1])
    tracker = TrackerMIL::create();
  else if (trackerType == trackerTypes[2])
    tracker = TrackerKCF::create();
  else if (trackerType == trackerTypes[3])
    tracker = TrackerTLD::create();
  else if (trackerType == trackerTypes[4])
    tracker = TrackerMedianFlow::create();
  else if (trackerType == trackerTypes[5])
    tracker = TrackerGOTURN::create();
  else if (trackerType == trackerTypes[6])
    tracker = TrackerMOSSE::create();
  else if (trackerType == trackerTypes[7])
    tracker = TrackerCSRT::create();
  else {
    cout << "Incorrect tracker name" << endl;
    cout << "Available trackers are: " << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
      std::cout << " " << *it << endl;
  }
  return tracker;
}

int main(int argc, char * argv[]) 
{
  cout << "Default tracking algoritm is CSRT" << endl;
  cout << "Available tracking algorithms are:" << endl;
  for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
    std::cout << " " << *it << endl;
  
  // set default values for tracking algorithm and video
  string videoPath = "videos/run.mp4";
  string trackingAlgo = "CSRT";
  
  // read videoPath and tracking Algo from arguments
  if (argc == 2) 
  {
    videoPath = argv[1];
  } else if (argc == 3) {
    videoPath = argv[1];
    trackingAlgo = argv[2];
  }

  
  string outputVideo = "results/multiTracker-" + trackingAlgo + ".avi";

  // Initialize MultiTracker with tracking algo
  Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();
  vector<Rect2d> objects;
  vector<Rect> ROIs;

  // create a video capture object to read videos
  cv::VideoCapture cap(videoPath);
  Mat frame;

  // quit if unabke to read video file
  if(!cap.isOpened()) {
    cout << "Error opening video file " << videoPath << endl;
    return -1;
  }

  int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  VideoWriter video(outputVideo,CV_FOURCC('M','J','P','G'), 25, Size(frame_width,frame_height));
  
  // read first frame
  cap >> frame;
  // draw bounding boxes over objects
  // selectROI's default behaviour is to draw box starting from the center
  // when fromCenter is set to false, you can draw box starting from top left corner
  bool showCrosshair = true;
  bool fromCenter = false;
  cout << "\n==========================================================\n";
  cout << "OpenCV says press c to cancel objects selection process" << endl;
  cout << "It doesn't work. Press Escape to exit selection process" << endl;
  cout << "\n==========================================================\n";
  cv::selectROIs("Image", frame, ROIs, showCrosshair, fromCenter);
  
  //ROIs.push_back(Rect(649, 234, 131, 269));
  //ROIs.push_back(Rect(976, 156, 184, 440));
  
  // quit if there are no objects to track
  if(ROIs.size() < 1)
    return 0;
  
  RNG rng(12345);
  vector<Scalar> colors;  

  // initialize the tracker
  std::vector<Ptr<Tracker> > trackers;
  for (int i = 0; i < ROIs.size(); i++) 
  {
    // List of trackers
    trackers.push_back(createTrackerByName(trackingAlgo));
    // Add bounding boxes
    objects.push_back(ROIs[i]);
    // Randomly choose tracker boxes
    colors.push_back(Scalar(rng.uniform(64,255), rng.uniform(64, 255), rng.uniform(64, 255))); 
  }

  // initialize multitracker
  multiTracker->add(trackers, frame, objects);
  
  // process video and track objects
  cout << "\n==========================================================\n";
  cout << "Started tracking, press ESC to quit." << endl;
  while(cap.isOpened()) 
  {
    // get frame from the video
    cap >> frame;
  
    // stop the program if reached end of video
    if (frame.empty()) break;

    //update the tracking result with new frame
    multiTracker->update(frame);
  
    // draw tracked objects
    for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
    {
      rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
    }
  
    // show frame
    //resize(frame, writeframe, Size(), 0.5, 0.5);
    writeframe = frame; 
    imshow("image", writeframe);
    video.write(writeframe);
  
    // quit on x button
    int key = waitKey(1);
    if  (key == 27) {
      break;
    }
   }

  video.release();
}