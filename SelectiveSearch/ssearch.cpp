#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>

using namespace cv;
using namespace cv::ximgproc::segmentation;

static void help() {
    std::cout << std::endl <<
    "Usage:" << std::endl <<
    "./ssearch input_image (f|q)" << std::endl <<
    "f=fast, q=quality" << std::endl <<
    "Use l to display less rects, m to display more rects, q to quit" << std::endl;
}


int main(int argc, char** argv) {
    // If image path and f/q is not passed as command
    // line arguments, quit and display help message
    if (argc < 3) {
        help();
        return -1;
    }

    // speed-up using multithreads
    setUseOptimized(true);
    setNumThreads(4);

    // read image
    Mat im = imread(argv[1]);
    // resize image
    int newHeight = 200;
    int newWidth = im.cols*newHeight/im.rows;
    resize(im, im, Size(newWidth, newHeight));

    // create Selective Search Segmentation Object using default parameters
    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    // set input image on which we will run segmentation
    ss->setBaseImage(im);

    // Switch to fast but low recall Selective Search method
    if (argv[2][0] == 'f') {
        ss->switchToSelectiveSearchFast();
    }
    // Switch to high recall but slow Selective Search method
    else if (argv[2][0] == 'q') {
        ss->switchToSelectiveSearchQuality();
    } 
    // if argument is neither f nor q print help message
    else {
        help();
        return -2;
    }

    // run selective search segmentation on input image
    std::vector<Rect> rects;
    ss->process(rects);
    std::cout << "Total Number of Region Proposals: " << rects.size() << std::endl;

    // number of region proposals to show
    int numShowRects = 100;
    // increment to increase/decrease total number
    // of reason proposals to be shown
    int increment = 50;

    while(1) {
        // create a copy of original image
        Mat imOut = im.clone();

        // itereate over all the region proposals
        for(int i = 0; i < rects.size(); i++) {
            if (i < numShowRects) {
                rectangle(imOut, rects[i], Scalar(0, 255, 0));
            }
            else {
                break;
            }
        }

        // show output
        imshow("Output", imOut);

        // record key press
        int k = waitKey();

        // m is pressed
        if (k == 109) {
            // increase total number of rectangles to show by increment
            numShowRects += increment;
        }
        // l is pressed
        else if (k == 108 && numShowRects > increment) {
            // decrease total number of rectangles to show by increment
            numShowRects -= increment;
        }
        // q is pressed
        else if (k == 113) {
            break;
        }
    }
    return 0;
}
