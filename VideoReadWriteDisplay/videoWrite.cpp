#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){

  // Create a VideoCapture object and use camera to capture the video
  VideoCapture cap(0); 

  // Check if camera opened successfully
  if(!cap.isOpened()){
   	cout << "Error opening video stream" << endl;
        return -1;
  }
  
  // Default resolutions of the frame are obtained.The default resolutions are system dependent.
  int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  
  // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
  VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height));

  while(1){

    Mat frame;
   
    // Capture frame-by-frame
    cap >> frame;
 
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
    
    // Write the frame into the file 'outcpp.avi'
    video.write(frame);
   
    // Display the resulting frame    
    imshow( "Frame", frame );
 
    // Press  ESC on keyboard to  exit
    char c = (char)waitKey(1);
    if( c == 27 ) 
      break;
  }

  // When everything done, release the video capture and write object
  cap.release();
  video.release();

  // Closes all the frames
  destroyAllWindows();
  return 0;
}
