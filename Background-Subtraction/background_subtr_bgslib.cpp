#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <opencv2/videoio.hpp>

#include "algorithms/algorithms.h"

using namespace cv;
using namespace std;

const char* input_params = "{ input | space_traffic.mp4 | Define the full input video path }";

void get_bgslib_result(String video_to_process)
{
    // create VideoCapture object for further video processing
    VideoCapture capture(samples::findFile(video_to_process));
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open: " << video_to_process << endl;
        return;
    }

    // instantiate background subtraction model
    SuBSENSE background_subtr_method = SuBSENSE();

    Mat frame, fgMask, background;
    while (true) {
        capture >> frame;

        // check whether the frames have been grabbed
        if (frame.empty())
            break;

        // resize video frames
        resize(frame, frame, Size(640, 360));

        try
        {
            // pass the frame to the model processor
            background_subtr_method.process(frame, fgMask, background);
        }
        catch (exception& e)
        {
            cout << "Exception occurred" << endl;
            cout << e.what() << endl;
        }

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
    get_bgslib_result(parser.get<String>("input"));

    return 0;
}