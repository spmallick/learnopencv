#include "opencv2/opencv.hpp" 

using namespace cv;
using namespace std;

// Specifying minimum and maximum size parameters
#define MIN_FACE_SIZE 100
#define MAX_FACE_SIZE 300

int main(void){
  // Load the Cascade Classifier Xml file
  CascadeClassifier faceCascade("cascade/mallick_haarcascade_frontalface_default.xml");
  
  // Create a VideoCapture object
  VideoCapture cap;

  // Check if camera opened successfully
  if(!cap.open(0)) return 0;
  
  Mat frame, frameBig, frameGray;
   
  while (1){
    
    // Reading each frame
    bool frameRead = cap.read(frameBig);

    // If frame not opened successfully
    if (!frameRead)
      break;

    // Fixing the scaling factor
    float scale = 640.0f/frameBig.cols;

    // Resizing the image
    resize(frameBig, frame, Size(), scale, scale);

    // Converting to grayscale
    cvtColor(frame, frameGray, COLOR_BGR2GRAY);

    // Creating vector to store the detected faces' parameters
    vector<Rect> faces;

    // Detect faces
    faceCascade.detectMultiScale(frameGray,faces,1.1,5,0,Size(MIN_FACE_SIZE,MIN_FACE_SIZE),Size(MAX_FACE_SIZE,MAX_FACE_SIZE));
    
    // Loop over each detected face
    for ( int i = 0; i < faces.size(); i++)
    {
      // Dimension parameters for bounding rectangle for face
      Rect faceRect = faces[i];

      // Calculating the dimension parameters for eyes from the dimensions parameters of the face
      Rect eyesRect = Rect(faceRect.x + 0.125*faceRect.width,faceRect.y + 0.25 * faceRect.height,0.75 * faceRect.width,
                       0.25 * faceRect.height);
      
      // Drawing the bounding rectangle around the face
      rectangle(frame, eyesRect,Scalar(128,255,0), 2);
      
    }
    
    // Display the resulting frame    
    imshow("Ninja Eye Detector", frame);
    int k = waitKey(1);

    // Press ESC on keyboard to stop tracking
    if(k == 27)
      break;
  }
  // release the VideoCapture object
  cap.release();
  
  // Closes all the windows
  destroyAllWindows();
  return 0;
}

