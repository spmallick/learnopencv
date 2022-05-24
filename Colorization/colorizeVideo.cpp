// This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project.
// It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example: ./colorizeVideo.out greyscaleVideo.mp4

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// the 313 ab cluster centers from pts_in_hull.npy (already transposed)
static float hull_pts[] = {
    -90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
    -70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
    -50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
    -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
    -20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
    -10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
    10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
    20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
    40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
    50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
    60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
    80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
    90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
    20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
    60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
    30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
    -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
    30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
    100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
    -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
    0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
    40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
    80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
    -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
    -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
    -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};

int main(int argc, char **argv)
{

    string videoFileName;
    string device;

    // Take arguments from command line
    if (argc == 3)
    {
        device = argv[2];
    }
    else if (argc == 2)
        device = "cpu";
    else
    {
        cout << "Please input the greyscale video filename." << endl;
        cout << "Usage example: ./colorizeVideo.out greyscaleVideo.mp4" << endl;
        cout << "If you want to use GPU device instead of CPU, add one more argument." << endl;
        cout << "Usage example: ./colorizeVideo.out greyscaleVideo.mp4 gpu" << endl;
        return 1;
    }
    videoFileName = argv[1];

    cv::VideoCapture cap(videoFileName);
    if (!cap.isOpened())
    {
        cerr << "Unable to open video" << endl;
        return 1;
    }

    cout << "Input video file: " << videoFileName << endl;

    string protoFile = "./models/colorization_deploy_v2.prototxt";
    string weightsFile = "./models/colorization_release_v2.caffemodel";

    Mat frame, frameCopy;
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    string str = videoFileName;
    str.replace(str.end() - 4, str.end(), "");
    string outVideoFileName = str + "_colorized.avi";
    VideoWriter video(outVideoFileName, VideoWriter::fourcc('M','J','P','G'), 60, Size(frameWidth,frameHeight));

    // fixed input size for the pre-trained network
    const int W_in = 224;
    const int H_in = 224;
    Net net = dnn::readNetFromCaffe(protoFile, weightsFile);
    if (device != "gpu")
    {
        cout << "Using CPU device" << endl;
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else
    {
        cout << "Using GPU device" << endl;
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }

    // setup additional layers:
    int sz[] = {2, 313, 1, 1};
    const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
    Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
    class8_ab->blobs.push_back(pts_in_hull);
    Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
    conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));

    vector<float> timer;

    for(;;)
    {

        cap >> frame;
        if (frame.empty()) break;
        
        frameCopy = frame.clone();
        
        double t = (double) cv::getTickCount();

        // extract L channel and subtract mean
        Mat lab, L, input;
        frame.convertTo(frame, CV_32F, 1.0/255);
        cvtColor(frame, lab, COLOR_BGR2Lab);
        extractChannel(lab, L, 0);
        resize(L, input, Size(W_in, H_in));
        input -= 50;
        
        // run the L channel through the network
        Mat inputBlob = blobFromImage(input);
        net.setInput(inputBlob);
        Mat result = net.forward();
        
        // retrieve the calculated a,b channels from the network output
        Size out_size(result.size[2], result.size[3]);
        Mat a = Mat(out_size, CV_32F, result.ptr(0, 0));
        Mat b = Mat(out_size, CV_32F, result.ptr(0, 1));

        resize(a, a, frame.size());
        resize(b, b, frame.size());
        
        // merge, and convert back to BGR
        Mat coloredFrame, chn[] = {L, a, b};
        merge(chn, 3, lab);
        cvtColor(lab, coloredFrame, COLOR_Lab2BGR);
        
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        timer.push_back(t);

        coloredFrame = coloredFrame.mul(255);
        coloredFrame.convertTo(coloredFrame, CV_8U);
        video.write(coloredFrame);

    }

    cout << "Time taken : " << accumulate(timer.begin(), timer.end(), 0.0) << " secs" << endl;
    cout << "Colorized video saved as " << outVideoFileName << endl << "Done !!!" << endl;
    cap.release();
    video.release();

    return 0;
}
