#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/optflow.hpp>
#include "lucas-kanade.cpp"
#include "dense_optical_flow.cpp"
#include <sys/stat.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const string keys =
            "{ h help |      | print this help message }"
            "{ @video |  | path to image file }"
            "{ @method | | method to OF calcualtion }"
            "{ save | | save video frames }";
    CommandLineParser parser(argc, argv, keys);

    string filename = samples::findFile(parser.get<string>("@video"));
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    string method = parser.get<string>("@method");
    printf("%s %s", method.c_str(), "method is now working!");
    bool save = false;
    if (parser.has("save")){
        save = true;
        mkdir("optical_flow_frames", 0777);
    }
    bool to_gray = true;
    if (method == "lucaskanade")
    {
        lucas_kanade(filename, save);
    }
    else if (method == "lucaskanade_dense"){
        dense_optical_flow(filename, save, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);
    }
    else if (method == "farneback"){
        dense_optical_flow(filename, save, calcOpticalFlowFarneback, to_gray, 0.5, 3, 15, 3, 5, 1.2, 0);
    }
    else if (method == "rlof"){
        to_gray = false;
        dense_optical_flow(
                filename, save, optflow::calcOpticalFlowDenseRLOF, to_gray,
                Ptr<cv::optflow::RLOFOpticalFlowParameter>(), 1.f, Size(6,6),
                cv::optflow::InterpolationType::INTERP_EPIC, 128, 0.05f, 999.0f, 15, 100, true, 500.0f, 1.5f, false
                );
    }
    return 0;
}