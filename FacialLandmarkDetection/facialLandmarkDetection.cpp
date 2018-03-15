#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;

void renderFace(Mat &im, vector<Rect> &faces, vector< vector<Point2f> > &landmarks)
{
  for(int i = 0; i < faces.size(); i++)
  {
    for(int k = 0; k < landmarks[i].size(); k++)
    {
      cv::circle(im,landmarks[i][k],3,cv::Scalar(0,0,255),FILLED);
    }
  }
  
}

static bool detectFace(InputArray image, OutputArray faces, CascadeClassifier *cascade)
{
    Mat imGray;
    cvtColor(image, imGray, COLOR_BGR2GRAY);
    
    vector<Rect> faces_;
    cascade->detectMultiScale(imGray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(100, 100));
    Mat(faces_).copyTo(faces);
    return true;
}

int main(int argc,char** argv)
{
    string faceDetectorFilename("haarcascade_frontalface_alt2.xml"); 
    string landmarkDetectorFilename("lbfmodel.yaml");
  
    CascadeClassifier faceDetector(faceDetectorFilename);
  
    FacemarkLBF::Params params;
    Ptr<Facemark> facemark = FacemarkLBF::create(params);

    facemark->setFaceDetector((FN_FaceDetector)detectFace, &faceDetector);
    facemark->loadModel(landmarkDetectorFilename);
  
    Mat im;
    VideoCapture cam(0);
    vector<Rect> faces;
    int count = 0; 
    while(cam.read(im))
    {
      
      if (count % 10 == 0)
      {
        facemark->getFaces(im,faces);
        count = 0; 
      }
      
      vector< vector<Point2f> > landmarks;
      bool success = facemark->fit(im,faces,landmarks);
      if(success)
      {
        renderFace(im, faces, landmarks);
        imshow("Facial Landmark Detection",im);
        int key = waitKey(1);
        if (key == 27)
        {
          break;
        }
      }
    }
    return 0;
}
