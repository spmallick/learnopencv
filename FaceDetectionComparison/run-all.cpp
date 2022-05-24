#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#if(CV_MAJOR_VERSION >= 3)
# include <opencv2/dnn.hpp>

using namespace cv::dnn;
#endif

#if(CV_MAJOR_VERSION < 3)
# include <dlib/opencv.h>
# include <dlib/image_processing.h>
# include <dlib/dnn.h>
# include <dlib/data_io.h>
# include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
#endif

#include <boost/algorithm/string.hpp>

using namespace cv;
using namespace std;

/** Global variables */
String faceCascadePath;
CascadeClassifier faceCascade;

void detectFaceOpenCVHaar(CascadeClassifier faceCascade, Mat &frameOpenCVHaar, int inHeight=300, int inWidth=0)
{
    int frameHeight = frameOpenCVHaar.rows;
    int frameWidth = frameOpenCVHaar.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameOpenCVHaarSmall, frameGray;
    resize(frameOpenCVHaar, frameOpenCVHaarSmall, Size(inWidth, inHeight));
    cvtColor(frameOpenCVHaarSmall, frameGray, COLOR_BGR2GRAY);

    std::vector<Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        int x1 = (int)(faces[i].x * scaleWidth);
        int y1 = (int)(faces[i].y * scaleHeight);
        int x2 = (int)((faces[i].x + faces[i].width) * scaleWidth);
        int y2 = (int)((faces[i].y + faces[i].height) * scaleHeight);
        cv::rectangle(frameOpenCVHaar, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}

#if(CV_MAJOR_VERSION >= 3)
const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);


const std::string caffeConfigFile = "models/deploy.prototxt";
const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, string framework="caffe")
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    cv::Mat inputBlob;
    if (framework == "caffe")
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
    else
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),(int)(frameHeight/150.0), 4);
        }
    }
}
#endif

#if(CV_MAJOR_VERSION < 3)
void detectFaceDlibHog(frontal_face_detector hogFaceDetector, Mat &frameDlibHog, int inHeight=300, int inWidth=0)
{

    int frameHeight = frameDlibHog.rows;
    int frameWidth = frameDlibHog.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameDlibHogSmall;
    resize(frameDlibHog, frameDlibHogSmall, Size(inWidth, inHeight));

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(frameDlibHogSmall);

    // Detect faces in the image
    std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

    for ( size_t i = 0; i < faceRects.size(); i++ )
    {
        int x1 = (int)(faceRects[i].left() * scaleWidth);
        int y1 = (int)(faceRects[i].top() * scaleHeight);
        int x2 = (int)(faceRects[i].right() * scaleWidth);
        int y2 = (int)(faceRects[i].bottom() * scaleHeight);
        cv::rectangle(frameDlibHog, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}

// Network Definition
/////////////////////////////////////////////////////////////////////////////////////////////////////
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
/////////////////////////////////////////////////////////////////////////////////////////////////////

void detectFaceDlibMMOD(net_type mmodFaceDetector, Mat &frameDlibMmod, int inHeight=300, int inWidth=0)
{

    int frameHeight = frameDlibMmod.rows;
    int frameWidth = frameDlibMmod.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameDlibMmodSmall;
    resize(frameDlibMmod, frameDlibMmodSmall, Size(inWidth, inHeight));

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(frameDlibMmodSmall);
    matrix<rgb_pixel> dlibMatrix;
    assign_image(dlibMatrix, dlibIm);

    // Detect faces in the image
    std::vector<dlib::mmod_rect> faceRects = mmodFaceDetector(dlibMatrix);

    for ( size_t i = 0; i < faceRects.size(); i++ )
    {
        int x1 = (int)(faceRects[i].rect.left() * scaleWidth);
        int y1 = (int)(faceRects[i].rect.top() * scaleHeight);
        int x2 = (int)(faceRects[i].rect.right() * scaleWidth);
        int y2 = (int)(faceRects[i].rect.bottom() * scaleHeight);
        cv::rectangle(frameDlibMmod, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}
#endif

int main( int argc, const char** argv )
{
    faceCascadePath = "models/haarcascade_frontalface_default.xml";

    if(!faceCascade.load(faceCascadePath))
    {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

#if(CV_MAJOR_VERSION < 3)
    frontal_face_detector hogFaceDetector = get_frontal_face_detector();
    String mmodModelPath = "models/mmod_human_face_detector.dat";
    net_type mmodFaceDetector;
    deserialize(mmodModelPath) >> mmodFaceDetector;
#endif

    string videoFileName;
    string device;
    string framework;
    // Take arguments from command line
    if (argc == 4)
    {
        videoFileName = argv[1];
        device = argv[2];
        framework = argv[3];
    }
    else if (argc == 3)
    {
        videoFileName = argv[1];
        device = argv[2];
        framework = "caffe";
    }
    else if (argc == 2)
    {
        videoFileName = argv[1];
        device = "cpu";
        framework = "caffe";
    }
    else
    {
        videoFileName = "";
        device = "gpu";
        framework = "caffe";
    }

    boost::to_upper(device);
    cout << "OpenCV DNN Configuration:" << endl;
    cout << "Device - "<< device << endl;
    if (framework == "caffe")
        cout << "Framework - Caffe" << endl;
    else
        cout << "Framework - TensorFlow" << endl;
    if (videoFileName == "")
        cout << "No video found, using camera stream" << endl;
    else
        cout << "Video file - " << videoFileName << endl;

    Net net;

    if (framework == "caffe")
        net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    else
        net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);

#if (CV_MAJOR_VERSION >= 4)
    if (device == "CPU")
    {
        net.setPreferableBackend(DNN_TARGET_CPU);
        cout << "Device - "<< device << endl;
    }
    else
    {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        cout << "Device - "<< device << endl;
    }
#elif(CV_MAJOR_VERSION == 3)
    // OpenCV 3.4.x does not support GPU backend
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    device = "cpu";
    cout << "Device - "<< device << endl;
#endif

    cv::VideoCapture source;
    if (videoFileName != "")
        source.open(videoFileName);
    else
        source.open(0, CAP_V4L);

    Mat frame;

    int frame_count = 0;
    double tt_opencvHaar = 0;
    double tt_opencvDNN = 0;
    double tt_dlibHog = 0;
    double tt_dlibMmod = 0;

    namedWindow("Face Detection Comparison", WINDOW_NORMAL);
    double t = 0;
    while (true)
    {
        source >> frame;
        if(frame.empty())
            break;

        frame_count++;

        t = cv::getTickCount();
        Mat frameOpenCVHaar = frame.clone();
        detectFaceOpenCVHaar ( faceCascade, frameOpenCVHaar );
        tt_opencvHaar += ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        double fpsOpencvHaar = frame_count/tt_opencvHaar;
        putText(frameOpenCVHaar, format("OpenCV HAAR; FPS = %.2f",fpsOpencvHaar), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);

        Mat frameOpenCVDNN = frame.clone();
#if(CV_MAJOR_VERSION >= 3)
        t = cv::getTickCount();
        detectFaceOpenCVDNN (net, frameOpenCVDNN, framework);
        tt_opencvDNN += ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        double fpsOpencvDNN = frame_count/tt_opencvDNN;
        putText(frameOpenCVDNN, format("OpenCV DNN %s FPS = %.2f", device.c_str(), fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);
#else
        putText(frameOpenCVDNN, "OpenCV DNN NOT SUPPORTED", Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);
#endif

#if(CV_MAJOR_VERSION < 3)
        t = cv::getTickCount();
        Mat frameDlibHog = frame.clone();
        detectFaceDlibHog ( hogFaceDetector, frameDlibHog );
        tt_dlibHog += ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        double fpsDlibHog = frame_count/tt_dlibHog;
        putText(frameDlibHog, format("DLIB HoG; FPS = %.2f",fpsDlibHog), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);

        t = cv::getTickCount();
        Mat frameDlibMmod = frame.clone();
        detectFaceDlibMMOD ( mmodFaceDetector, frameDlibMmod );
        tt_dlibMmod += ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        double fpsDlibMmod = frame_count/tt_dlibMmod;
        putText(frameDlibMmod, format("DLIB MMOD; FPS = %.2f",fpsDlibMmod), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);
#endif

        Mat top, bottom, combined;
        hconcat(frameOpenCVHaar, frameOpenCVDNN, top);
#if(CV_MAJOR_VERSION < 3)
        hconcat(frameDlibHog, frameDlibMmod, bottom);
        vconcat(top, bottom, combined);
#else
        combined = top;
#endif

        cv::resize(combined, combined, Size(), .5, .5);
        imshow("Face Detection Comparison", combined);

        int k = waitKey(5);
        if(k == 27)
        {
          destroyAllWindows();
          break;
        }

        if(frame_count == 1)
        {
            tt_opencvHaar = 0;
            tt_opencvDNN = 0;
            tt_dlibHog = 0;
            tt_dlibMmod = 0;
        }
    }
}
