#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>
using namespace cv;
using namespace cv::dnn;
using namespace std;

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat &frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    // std::vector<int> meanVal = {104, 117, 123};
    Scalar meanVal = Scalar(104, 117, 123);

    cv::Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    vector<vector<int>> bboxes;

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = {x1, y1, x2, y2};
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

    return make_tuple(frameOpenCVDNN, bboxes);
}

int main(int argc, char** argv)
{
    string faceProto = "opencv_face_detector.pbtxt";
    string faceModel = "opencv_face_detector_uint8.pb";

    string ageProto = "age_deploy.prototxt";
    string ageModel = "age_net.caffemodel";

    string genderProto = "gender_deploy.prototxt";
    string genderModel = "gender_net.caffemodel";

    Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

    vector<string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
      "(38-43)", "(48-53)", "(60-100)"};

    vector<string> genderList = {"Male", "Female"};

    // Load Network
    Net ageNet = readNet(ageModel, ageProto);
    Net genderNet = readNet(genderModel, genderProto);
    Net faceNet = readNet(faceModel, faceProto);

    VideoCapture cap;
    if (argc > 1)
        cap.open(argv[1]);
    else
        cap.open(0);
    int padding = 20;
    while(waitKey(1) < 0) {
      // read frame
      Mat frame;
      cap.read(frame);
      if (frame.empty())
      {
          waitKey();
          break;
      }

      vector<vector<int>> bboxes;
      Mat frameFace;
      tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);

      if(bboxes.size() == 0) {
        cout << "No face detected, checking next frame." << endl;
        continue;
      }
      for (auto it = begin(bboxes); it != end(bboxes); ++it) {
        Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2*padding, it->at(3) - it->at(1) + 2*padding);
        Mat face = frame(rec); // take the ROI of box on the frame

        Mat blob;
        blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
        genderNet.setInput(blob);
        // string gender_preds;
        vector<float> genderPreds = genderNet.forward();
        // printing gender here
        // find max element index
        // distance function does the argmax() work in C++
        int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
        string gender = genderList[max_index_gender];
        cout << "Gender: " << gender << endl;

        /* // Uncomment if you want to iterate through the gender_preds vector
        for(auto it=begin(gender_preds); it != end(gender_preds); ++it) {
          cout << *it << endl;
        }
        */

        ageNet.setInput(blob);
        vector<float> agePreds = ageNet.forward();
        /* // uncomment below code if you want to iterate through the age_preds
         * vector
        cout << "PRINTING AGE_PREDS" << endl;
        for(auto it = age_preds.begin(); it != age_preds.end(); ++it) {
          cout << *it << endl;
        }
        */

        // finding maximum indicd in the age_preds vector
        int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
        string age = ageList[max_indice_age];
        cout << "Age: " << age << endl;
        string label = gender + ", " + age; // label
        cv::putText(frameFace, label, Point(it->at(0), it->at(1) -15), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
        imshow("Frame", frameFace);
        imwrite("out.jpg",frameFace);
      }

    }
}
