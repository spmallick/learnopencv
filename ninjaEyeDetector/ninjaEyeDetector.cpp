#include "opencv2/opencv.hpp" 

using namespace cv;
using namespace std;

#define MIN_FACE_SIZE 100
#define MAX_FACE_SIZE 300

int main(void)
{
  
  CascadeClassifier faceCascade("cascade/mallick_haarcascade_frontalface_default.xml");
  
  VideoCapture cap;
  if(!cap.open(0)) return 0;
  
  
  
  Mat frame, frameBig, frameGray;
  
  
  
  while (1)
  {
    bool frameRead = cap.read(frameBig);
    if (!frameRead) break;
    float scale = 640.0f/frameBig.cols;
    resize(frameBig, frame, Size(), scale, scale);
    cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    vector<Rect> faces;
    faceCascade.detectMultiScale(frameGray,
                                  faces,
                                  1.1,
                                  5,
                                  0,
                                  Size(MIN_FACE_SIZE,MIN_FACE_SIZE),
                                  Size(MAX_FACE_SIZE,MAX_FACE_SIZE)
                                  );
    
    
    for ( int i = 0; i < faces.size(); i++)
    {
      Rect faceRect = faces[i];
      Rect eyesRect = Rect(
                       faceRect.x + 0.125*faceRect.width,
                       faceRect.y + 0.25 * faceRect.height,
                       0.75 * faceRect.width,
                       0.25 * faceRect.height
                       );
      
      rectangle(frame, eyesRect,Scalar(128,255,0), 2);
      
    }
    
    imshow("Ninja Eye Detector", frame);
    int k = waitKey(1);
    if(k == 27)
      break;
  }
  
}
