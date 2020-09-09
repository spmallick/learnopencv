#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace cv;
using namespace std;


void calculate_optical_flow(string videoFileName, string device)
{
    // init map to track time for every stage at each iteration
    map<string, vector<double>> timers;

    // init video capture with video
    VideoCapture capture(videoFileName);
    if (!capture.isOpened())
    {
        // error in opening the video file
        cout << "Unable to open file!" << endl;
        return;
    }

    // get default video FPS
    double fps = capture.get(CAP_PROP_FPS);
    // get total number of video frames
    int num_frames = int(capture.get(CAP_PROP_FRAME_COUNT));

    // read the first frame
    Mat frame, previous_frame;
    capture >> frame;

    // resize frame
    resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);

    // convert to gray
    cvtColor(frame, previous_frame, COLOR_BGR2GRAY);

    // declare needed variables outside of loop
    Mat flow (previous_frame.size(), CV_32FC2);
    Mat flow_x, flow_y;
    Mat magnitude, angle, normalized_magnitude;
    Mat hsv[3], merged_hsv, hsv_u8, bgr;

    // create optical flow instance
    cuda::GpuMat gpu_flow;
    Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);


    while (true)
    {
        // start full pipeline timer
        auto start_full_time = chrono::high_resolution_clock::now();

        // start reading timer
        auto start_read_time = chrono::high_resolution_clock::now();

        // capture frame-by-frame
        capture >> frame;

        // end reading timer
        auto end_read_time = chrono::high_resolution_clock::now();
        // add elapsed iteration time
        timers["reading"].push_back(chrono::duration_cast<chrono::milliseconds>(end_read_time - start_read_time).count()/1000.0);

        if (frame.empty())
            break;

        // start pre-process timer
        auto start_pre_time = chrono::high_resolution_clock::now();

        // resize frame
        resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);

        // convert to gray
        Mat current_frame;
        cvtColor(frame, current_frame, COLOR_BGR2GRAY);

        if (device == "cpu")
        {
            // end pre-process timer
            auto end_pre_time = chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["pre-process"].push_back(chrono::duration_cast<chrono::milliseconds>(end_pre_time - start_pre_time).count()/1000.0);

            // start optical flow timer
            auto start_of_time = chrono::high_resolution_clock::now();

            // calculate optical flow
            calcOpticalFlowFarneback(previous_frame, current_frame, flow, 0.5, 5, 15, 3, 5, 1.2, 0);

            // end optical flow timer
            auto end_of_time = chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["optical flow"].push_back(chrono::duration_cast<chrono::milliseconds>(end_of_time - start_of_time).count()/1000.0);

            // split the output flow into 2 vectors
            Mat planes[2];
            split(flow, planes);

            // get the result
            flow_x = planes[0];
            flow_y = planes[1];
        }
        else
        {
            // move both frames to GPU
            cuda::GpuMat cu_previous (previous_frame);
            cuda::GpuMat cu_current (current_frame);

            // end pre-process timer
            auto end_pre_time = chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["pre-process"].push_back(chrono::duration_cast<chrono::milliseconds>(end_pre_time - start_pre_time).count()/1000.0);

            // start optical flow timer
            auto start_of_time = chrono::high_resolution_clock::now();

            // calculate optical flow
            ptr_calc->calc(cu_previous, cu_current, gpu_flow);

            // end optical flow timer
            auto end_of_time = chrono::high_resolution_clock::now();

            // add elapsed iteration time
            timers["optical flow"].push_back(chrono::duration_cast<chrono::milliseconds>(end_of_time - start_of_time).count()/1000.0);

            // split the output flow into 2 vectors
            cuda::GpuMat planes[2];
            cuda::split(gpu_flow, planes);

            // send result from GPU back to CPU
            planes[0].download(flow_x);
            planes[1].download(flow_y);
        }

        // start post-process timer
        auto start_post_time = chrono::high_resolution_clock::now();

        // convert from cartesian to polar coordinates
        cartToPolar(flow_x, flow_y, magnitude, angle, true);

        // normalize magnitude from 0 to 1
        normalize(magnitude, normalized_magnitude, 0.0f, 1.0f, NORM_MINMAX);

        // get angle of optical flow
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        // build hsv image
        hsv[0] = angle;
        hsv[1] = Mat::ones(angle.size(), CV_32F);
        hsv[2] = normalized_magnitude;
        merge(hsv, 3, merged_hsv);

        // multiply each pixel value to 255
        merged_hsv.convertTo(hsv_u8, CV_8U, 255.0);

        // convert hsv to bgr
        cvtColor(hsv_u8, bgr, COLOR_HSV2BGR);

        // end post pipeline timer
        auto end_post_time = chrono::high_resolution_clock::now();

        // add elapsed iteration time
        timers["post-process"].push_back(chrono::duration_cast<chrono::milliseconds>(end_post_time - start_post_time).count()/1000.0);

        // end full pipeline timer
        auto end_full_time = chrono::high_resolution_clock::now();

        // add elapsed iteration time
        timers["full pipeline"].push_back(chrono::duration_cast<chrono::milliseconds>(end_full_time - start_full_time).count()/1000.0);

        // visualization
        imshow("original", frame);
        imshow("result", bgr);
        int keyboard = waitKey(1);
        if (keyboard == 27)
            break;

        // update previous_frame value
        previous_frame = current_frame;
    }

    // print results
    cout << "Number of frames: "   << num_frames << std::endl;

    // elapsed time at each stage
    cout << "Elapsed time"   << num_frames << std::endl;

    for (auto const& timer : timers)
    {
        cout << "- " << timer.first << " : " << accumulate(timer.second.begin(), timer.second.end(), 0.0) << " seconds"<< endl;
    }

    // calculate frames per second
    cout << "Default video FPS : "  << fps << endl;
    float optical_flow_fps  = (num_frames - 1) / accumulate(timers["optical flow"].begin(),  timers["optical flow"].end(),  0.0);
    cout << "Optical flow FPS : "   << optical_flow_fps  << endl;

    float full_pipeline_fps = (num_frames - 1) / accumulate(timers["full pipeline"].begin(), timers["full pipeline"].end(), 0.0);
    cout << "Full pipeline FPS : "  << full_pipeline_fps << endl;
}

int main( int argc, const char** argv )
{
    string videoFileName;
    string device;

    // parse arguments from command line
    if (argc == 3)
    {
        videoFileName = argv[1];
        device = argv[2];
    }
    else if (argc == 2)
    {
        videoFileName = argv[1];
        device = "cpu";
    }
    else
    {
        cout << "Please input video filename." << endl;
        cout << "Usage example: ./demo.out video/boat.mp4" << endl;
        cout << "If you want to use GPU device instead of CPU, add one more argument." << endl;
        cout << "Usage example: ./demo.out video/boat.mp4 gpu" << endl;
        return 1;
    }

    // output passed arguments
    cout << "Configuration" << endl;
    cout << "- device : "<< device << endl;
    cout << "- video file : " << videoFileName << endl;

    calculate_optical_flow(videoFileName, device);

    return 0;
}

