// This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:   ./augmented_reality_with_aruco.out --image=test.jpg
//                  ./augmented_reality_with_aruco.out --video=test.mp4
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./augmented_reality_with_aruco.out --image=test.jpg \n\t\t./augmented_reality_with_aruco.out --video=test.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace aruco;
using namespace std;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to do Augmented Reality using Aruco markers in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    Mat im_src = imread("new_scenery.jpg");

    try {
        
        outputFile = "ar_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_ar_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_ar_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcaom
        else cap.open(parser.get<int>("device"));
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(2*cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
    
    // Create a window
    static const string kWinName = "Augmented Reality using Aruco markers in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;
        
        try {
            // Stop the program if reached end of video
            if (frame.empty()) {
                cout << "Done processing !!!" << endl;
                cout << "Output file is stored as " << outputFile << endl;
                waitKey(3000);
                break;
            }

            vector<int> markerIds;
            
            // Load the dictionary that was used to generate the markers.
            Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_6X6_250);

            // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
            vector<vector<Point2f>> markerCorners, rejectedCandidates;

            // Initialize the detector parameters using default values
            Ptr<DetectorParameters> parameters = DetectorParameters::create();

            // Detect the markers in the image
            detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

            // Using the detected markers, locate the quadrilateral on the target frame where the new scene is going to be displayed.
            vector<Point> pts_dst;
            float scalingFac = 0.02;//0.015;

            Point refPt1, refPt2, refPt3, refPt4;

            // finding top left corner point of the target quadrilateral
            std::vector<int>::iterator it = std::find(markerIds.begin(), markerIds.end(), 25);
            int index = std::distance(markerIds.begin(), it);
            refPt1 = markerCorners.at(index).at(1);

            // finding top right corner point of the target quadrilateral
            it = std::find(markerIds.begin(), markerIds.end(), 33);
            index = std::distance(markerIds.begin(), it);
            refPt2 = markerCorners.at(index).at(2);
            
            float distance = norm(refPt1-refPt2);
            pts_dst.push_back(Point(refPt1.x - round(scalingFac*distance), refPt1.y - round(scalingFac*distance)));
            
            pts_dst.push_back(Point(refPt2.x + round(scalingFac*distance), refPt2.y - round(scalingFac*distance)));

            // finding bottom right corner point of the target quadrilateral
            it = std::find(markerIds.begin(), markerIds.end(), 30);
            index = std::distance(markerIds.begin(), it);
            refPt3 = markerCorners.at(index).at(0);
            pts_dst.push_back(Point(refPt3.x + round(scalingFac*distance), refPt3.y + round(scalingFac*distance)));

            // finding bottom left corner point of the target quadrilateral
            it = std::find(markerIds.begin(), markerIds.end(), 23);
            index = std::distance(markerIds.begin(), it);
            refPt4 = markerCorners.at(index).at(0);
            pts_dst.push_back(Point(refPt4.x - round(scalingFac*distance), refPt4.y + round(scalingFac*distance)));

            // Get the corner points of the new scene image.
            vector<Point> pts_src;
            pts_src.push_back(Point(0,0));
            pts_src.push_back(Point(im_src.cols, 0));
            pts_src.push_back(Point(im_src.cols, im_src.rows));
            pts_src.push_back(Point(0, im_src.rows));

            // Compute homography from source and destination points
            Mat h = cv::findHomography(pts_src, pts_dst);

            // Warped image
            Mat warpedImage;
            
            // Warp source image to destination based on homography
            warpPerspective(im_src, warpedImage, h, frame.size(), INTER_CUBIC);
        
            // Prepare a mask representing region to copy from the warped image into the original frame.
            Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            fillConvexPoly(mask, pts_dst, Scalar(255, 255, 255), LINE_AA);
            
            // Erode the mask to not copy the boundary effects from the warping
            Mat element = getStructuringElement( MORPH_RECT, Size(5,5));
//            Mat element = getStructuringElement( MORPH_RECT, Size(3,3));
            erode(mask, mask, element);

            // Copy the warped image into the original frame in the mask region.
            Mat imOut = frame.clone();
            warpedImage.copyTo(imOut, mask);
            
            // Showing the original image and the new output image side by side
            Mat concatenatedOutput;
            hconcat(frame, imOut, concatenatedOutput);

            if (parser.has("image")) imwrite(outputFile, concatenatedOutput);
            else video.write(concatenatedOutput);

            imshow(kWinName, concatenatedOutput);
            
        }
        catch(const std::exception& e) {
            cout << endl << " e : " << e.what() << endl;
            cout << "Could not do homography !! " << endl;
    //        return 0;
        }

    }

    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}
