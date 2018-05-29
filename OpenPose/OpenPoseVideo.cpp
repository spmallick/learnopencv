#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define MPI

#ifdef MPI
const int POSE_PAIRS[14][2] = 
{   
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13}
};

string protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt";
string weightsFile = "pose/mpi/pose_iter_160000.caffemodel";

int nPoints = 15;
#endif

#ifdef COCO
const int POSE_PAIRS[17][2] = 
{   
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
};

string protoFile = "pose/coco/pose_deploy_linevec.prototxt";
string weightsFile = "pose/coco/pose_iter_440000.caffemodel";

int nPoints = 18;
#endif

int main(int argc, char **argv)
{

    cout << "USAGE : ./openpose <VideoFile> " << endl;
    
    string videoFile = "sample_video.mp4";
    // Take arguments from commmand line
    if (argc == 2)
    {   
      videoFile = argv[1];
    }

    int inWidth = 368;
    int inHeight = 368;
    float thresh = 0.01;    

    cv::VideoCapture cap(videoFile);

    if (!cap.isOpened())
    {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }
    
    Mat frame, frameCopy;
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    VideoWriter video("Output-Skeleton.avi",VideoWriter::fourcc('M','J','P','G'), 10, Size(frameWidth,frameHeight));

    Net net = readNetFromCaffe(protoFile, weightsFile);
    double t=0;
    while( waitKey(1) < 0)
    {       
        double t = (double) cv::getTickCount();

        cap >> frame;
        frameCopy = frame.clone();
        Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

        net.setInput(inpBlob);

        Mat output = net.forward();

        int H = output.size[2];
        int W = output.size[3];

        // find the position of the body parts
        vector<Point> points(nPoints);
        for (int n=0; n < nPoints; n++)
        {
            // Probability map of corresponding body's part.
            Mat probMap(H, W, CV_32F, output.ptr(0,n));

            Point2f p(-1,-1);
            Point maxLoc;
            double prob;
            minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
            if (prob > thresh)
            {
                p = maxLoc;
                p.x *= (float)frameWidth / W ;
                p.y *= (float)frameHeight / H ;

                circle(frameCopy, cv::Point((int)p.x, (int)p.y), 8, Scalar(0,255,255), -1);
                cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1.1, cv::Scalar(0, 0, 255), 2);
            }
            points[n] = p;
        }

        int nPairs = sizeof(POSE_PAIRS)/sizeof(POSE_PAIRS[0]);

        for (int n = 0; n < nPairs; n++)
        {
            // lookup 2 connected body/hand parts
            Point2f partA = points[POSE_PAIRS[n][0]];
            Point2f partB = points[POSE_PAIRS[n][1]];

            if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
                continue;

            line(frame, partA, partB, Scalar(0,255,255), 8);
            circle(frame, partA, 8, Scalar(0,0,255), -1);
            circle(frame, partB, 8, Scalar(0,0,255), -1);
        }

        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cv::putText(frame, cv::format("time taken = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
        // imshow("Output-Keypoints", frameCopy);
        imshow("Output-Skeleton", frame);
        video.write(frame);
    }
    // When everything done, release the video capture and write object
    cap.release();
    video.release();

    return 0;
}