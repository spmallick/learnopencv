#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void verify_func(Mat* img);

int main(int argc, char** argv){
    Mat img = imread(argv[1]);

    imshow("img", img);
    waitKey(0);
    destroyAllWindows();

    verify_func(&img);

    imshow("img", img);
    waitKey(0);
    destroyAllWindows();
}

void verify_func(Mat* img){ 
    resize(*img, *img, Size(100, 100), 0, 0, INTER_CUBIC);
}
