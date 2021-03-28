#include <iostream>
#include <sstream>
#include <opencv2/bgsegm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;
using namespace cv::bgsegm;

const char* input_params = "{ input | space_traffic.mp4 | Define the full input video path }";

void get_opencv_result(String video_to_process) {
    // create VideoCapture object for further video processing
    VideoCapture capture(samples::findFile(video_to_process));
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open: " << video_to_process << endl;
        return;
    }

    // instantiate background subtraction model
    Ptr<BackgroundSubtractorGSOC> background_subtr_method = createBackgroundSubtractorGSOC();

    Mat frame, fgMask, background;
    while (true) {
        capture >> frame;

        // check whether the frames have been grabbed
        if (frame.empty())
            break;

        // resize video frames
        resize(frame, frame, Size(640, 360));

        // pass the frame to the background subtractor
        background_subtr_method->apply(frame, fgMask);
        // obtain the background without foreground mask
        background_subtr_method->getBackgroundImage(background);

        // show the current frame, foreground mask, subtracted result
        imshow("Initial Frames", frame);
        imshow("Foreground Masks", fgMask);
        imshow("Subtraction Result", background);

        int keyboard = waitKey(10);
        if (keyboard == 27)
            break;
    }
}


int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, input_params);
    // start BS-pipeline
    get_opencv_result(parser.get<String>("input"));

    return 0;
}