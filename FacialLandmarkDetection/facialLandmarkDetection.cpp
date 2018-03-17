#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "renderFace.hpp"

#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;


static bool detectFace(InputArray image, OutputArray faces, CascadeClassifier *cascade)
{
    // Convert image to grayscale
    Mat imGray;
    cvtColor(image, imGray, COLOR_BGR2GRAY);
    
    // Variable for storing face rectangles
    vector<Rect> facesRects;

    // Run OpenCV HAAR based face detector
    cascade->detectMultiScale(imGray, facesRects, 1.5, 5, CASCADE_SCALE_IMAGE, Size(100, 100));
    
    // Copy results to output array 
    Mat(facesRects).copyTo(faces);
    return true;
}

int main(int argc,char** argv)
{
    // Load Face Detector
    CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

    // Create an instance of FacemarkLBF. 
    // The optional argument params can be used to read the parameters
    // of the trained model 
    FacemarkLBF::Params params;
    Ptr<Facemark> facemark = FacemarkLBF::create(params);

    // Set face detector 
    facemark->setFaceDetector((FN_FaceDetector)detectFace, &faceDetector);
    
    // Load landmark detector
    facemark->loadModel("lbfmodel.yaml");
  
    // Set up webcam for video capture
    VideoCapture cam(0);
    
    // Variable to store a video frame
    Mat frame;

    // Variable to store detected face rectangles
    vector<Rect> faces;
    
    int count = 0; 

    // Read a frame
    while(cam.read(frame))
    {
      // Run face detection every 10th frame. 
      // This is done for speedup
      if (count % 10 == 0)
      {
        faces.clear();
        facemark->getFaces(frame,faces);
        count = 0; 
      }
      
      // Variable for landmarks. 
      // Landmarks for one face is a vector of points
      // There can be more than one face. Hence, we 
      // use a vector of vector of points. 
      vector< vector<Point2f> > landmarks;

      // Run landmark detector
      bool success = facemark->fit(frame,faces,landmarks);
      if(success)
      {
        // If successful, render the landmarks on the face
        for(int i = 0; i < landmarks.size(); i++)
        {
          renderFace(frame, landmarks[i]);
        }
        // Display results 
        imshow("Facial Landmark Detection", frame);
        int key = waitKey(1);

        // Exit loop if ESC is pressed
        if (key == 27)
        {
          break;
        }
      }
    }
    return 0;
}
