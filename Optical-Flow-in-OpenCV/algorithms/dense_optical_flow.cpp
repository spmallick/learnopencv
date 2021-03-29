#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


using namespace cv;
using namespace std;

template <typename Method, typename... Args>
void dense_optical_flow(string filename, bool save, Method method, bool to_gray, Args&&... args)
{
    VideoCapture capture(samples::findFile(filename));
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
    }
    Mat frame1, prvs;
    capture >> frame1;
    if (to_gray)
        cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    else
        prvs = frame1;
    int counter = 0;
    while (true) {
        Mat frame2, next;
        capture >> frame2;
        if (frame2.empty())
            break;
        if (to_gray)
            cvtColor(frame2, next, COLOR_BGR2GRAY);
        else
            next = frame2;
        Mat flow(prvs.size(), CV_32FC2);
        method(prvs, next, flow, std::forward<Args>(args)...);
        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
        //build hsv image
        Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);
        if (save) {
            string save_path = "./optical_flow_frames/frame_" + to_string(counter) + ".jpg";
            imwrite(save_path, bgr);
        }
        imshow("frame", frame2);
        imshow("flow", bgr);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        prvs = next;
        counter++;
    }
}

