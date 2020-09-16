#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace std;
using namespace std::chrono;


void calculate_optical_flow(string videoFileName, string device)
{
    // init map to track time for every stage at each iteration
    unordered_map<string, vector<double>> timers;

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
    cv::Mat frame, previous_frame;
    capture >> frame;

    if (device == "cpu")
    {
        // resize frame
        cv::resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);

        // convert to gray
        cv::cvtColor(frame, previous_frame, COLOR_BGR2GRAY);

        // declare outputs for optical flow
        cv::Mat magnitude, normalized_magnitude, angle;
        cv::Mat hsv[3], merged_hsv, hsv_8u, bgr;

        // set saturation to 1
        hsv[1] = cv::Mat::ones(frame.size(), CV_32F);

        while (true)
        {
            // start full pipeline timer
            auto start_full_time = high_resolution_clock::now();

            // start reading timer
            auto start_read_time = high_resolution_clock::now();

            // capture frame-by-frame
            capture >> frame;

            if (frame.empty())
                break;

            // end reading timer
            auto end_read_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["reading"].push_back(duration_cast<milliseconds>(end_read_time - start_read_time).count() / 1000.0);

            // start pre-process timer
            auto start_pre_time = high_resolution_clock::now();

            // resize frame
            cv::resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);

            // convert to gray
            cv::Mat current_frame;
            cv::cvtColor(frame, current_frame, COLOR_BGR2GRAY);

            // end pre-process timer
            auto end_pre_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["pre-process"].push_back(duration_cast<milliseconds>(end_pre_time - start_pre_time).count() / 1000.0);

            // start optical flow timer
            auto start_of_time = high_resolution_clock::now();

            // calculate optical flow
            cv::Mat flow;
            calcOpticalFlowFarneback(previous_frame, current_frame, flow, 0.5, 5, 15, 3, 5, 1.2, 0);

            // end optical flow timer
            auto end_of_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["optical flow"].push_back(duration_cast<milliseconds>(end_of_time - start_of_time).count() / 1000.0);

            // start post-process timer
            auto start_post_time = high_resolution_clock::now();

            // split the output flow into 2 vectors
            cv::Mat flow_xy[2], flow_x, flow_y;
            split(flow, flow_xy);

            // get the result
            flow_x = flow_xy[0];
            flow_y = flow_xy[1];

            // convert from cartesian to polar coordinates
            cv::cartToPolar(flow_x, flow_y, magnitude, angle, true);

            // normalize magnitude from 0 to 1
            cv::normalize(magnitude, normalized_magnitude, 0.0, 1.0, NORM_MINMAX);

            // get angle of optical flow
            angle *= ((1 / 360.0) * (180 / 255.0));

            // build hsv image
            hsv[0] = angle;
            hsv[2] = normalized_magnitude;
            merge(hsv, 3, merged_hsv);

            // multiply each pixel value to 255
            merged_hsv.convertTo(hsv_8u, CV_8U, 255);

            // convert hsv to bgr
            cv::cvtColor(hsv_8u, bgr, COLOR_HSV2BGR);

            // update previous_frame value
            previous_frame = current_frame;

            // end post pipeline timer
            auto end_post_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["post-process"].push_back(duration_cast<milliseconds>(end_post_time - start_post_time).count() / 1000.0);

            // end full pipeline timer
            auto end_full_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["full pipeline"].push_back(duration_cast<milliseconds>(end_full_time - start_full_time).count() / 1000.0);

            // visualization
            imshow("original", frame);
            imshow("result", bgr);
            int keyboard = waitKey(1);
            if (keyboard == 27)
                break;
        }
    }
    else
    {
        // resize frame
        cv::resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);

        // convert to gray
        cv::cvtColor(frame, previous_frame, COLOR_BGR2GRAY);

        // upload pre-processed frame to GPU
        cv::cuda::GpuMat gpu_previous;
        gpu_previous.upload(previous_frame);

        // declare cpu outputs for optical flow
        cv::Mat hsv[3], angle, bgr;

        // declare gpu outputs for optical flow
        cv::cuda::GpuMat gpu_magnitude, gpu_normalized_magnitude, gpu_angle;
        cv::cuda::GpuMat gpu_hsv[3], gpu_merged_hsv, gpu_hsv_8u, gpu_bgr;

        // set saturation to 1
        hsv[1] = cv::Mat::ones(frame.size(), CV_32F);
        gpu_hsv[1].upload(hsv[1]);

        while (true)
        {
            // start full pipeline timer
            auto start_full_time = high_resolution_clock::now();

            // start reading timer
            auto start_read_time = high_resolution_clock::now();

            // capture frame-by-frame
            capture >> frame;

            if (frame.empty())
                break;

            // upload frame to GPU
            cv::cuda::GpuMat gpu_frame;
            gpu_frame.upload(frame);

            // end reading timer
            auto end_read_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["reading"].push_back(duration_cast<milliseconds>(end_read_time - start_read_time).count() / 1000.0);

            // start pre-process timer
            auto start_pre_time = high_resolution_clock::now();

            // resize frame
            cv::cuda::resize(gpu_frame, gpu_frame, Size(960, 540), 0, 0, INTER_LINEAR);

            // convert to gray
            cv::cuda::GpuMat gpu_current;
            cv::cuda::cvtColor(gpu_frame, gpu_current, COLOR_BGR2GRAY);

            // end pre-process timer
            auto end_pre_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["pre-process"].push_back(duration_cast<milliseconds>(end_pre_time - start_pre_time).count() / 1000.0);

            // start optical flow timer
            auto start_of_time = high_resolution_clock::now();

            // create optical flow instance
            Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
            // calculate optical flow
            cv::cuda::GpuMat gpu_flow;
            ptr_calc->calc(gpu_previous, gpu_current, gpu_flow);

            // end optical flow timer
            auto end_of_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["optical flow"].push_back(duration_cast<milliseconds>(end_of_time - start_of_time).count() / 1000.0);

            // start post-process timer
            auto start_post_time = high_resolution_clock::now();

            // split the output flow into 2 vectors
            cv::cuda::GpuMat gpu_flow_xy[2];
            cv::cuda::split(gpu_flow, gpu_flow_xy);

            // convert from cartesian to polar coordinates
            cv::cuda::cartToPolar(gpu_flow_xy[0], gpu_flow_xy[1], gpu_magnitude, gpu_angle, true);

            // normalize magnitude from 0 to 1
            cv::cuda::normalize(gpu_magnitude, gpu_normalized_magnitude, 0.0, 1.0, NORM_MINMAX, -1);

            // get angle of optical flow
            gpu_angle.download(angle);
            angle *= ((1 / 360.0) * (180 / 255.0));

            // build hsv image
            gpu_hsv[0].upload(angle);
            gpu_hsv[2] = gpu_normalized_magnitude;
            cv::cuda::merge(gpu_hsv, 3, gpu_merged_hsv);

            // multiply each pixel value to 255
            gpu_merged_hsv.cv::cuda::GpuMat::convertTo(gpu_hsv_8u, CV_8U, 255.0);

            // convert hsv to bgr
            cv::cuda::cvtColor(gpu_hsv_8u, gpu_bgr, COLOR_HSV2BGR);

            // send original frame from GPU back to CPU
            gpu_frame.download(frame);

            // send result from GPU back to CPU
            gpu_bgr.download(bgr);

            // update previous_frame value
            gpu_previous = gpu_current;

            // end post pipeline timer
            auto end_post_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["post-process"].push_back(duration_cast<milliseconds>(end_post_time - start_post_time).count() / 1000.0);

            // end full pipeline timer
            auto end_full_time = high_resolution_clock::now();

            // add elapsed iteration time
            timers["full pipeline"].push_back(duration_cast<milliseconds>(end_full_time - start_full_time).count() / 1000.0);

            // visualization
            imshow("original", frame);
            imshow("result", bgr);
            int keyboard = waitKey(1);
            if (keyboard == 27)
                break;
        }

    }

    // release the capture
    capture.release();

    // destroy all windows
    destroyAllWindows();

    // print results
    cout << "Number of frames: "   << num_frames << std::endl;

    // elapsed time at each stage
    cout << "Elapsed time" << std::endl;
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

