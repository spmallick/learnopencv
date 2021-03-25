#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    /*
    Contour detection and drawing using different extraction modes to complement 
    the understanding of hierarchies
    */
    Mat image2 = imread("../../input/custom_colors.jpg");
    Mat img_gray2;
    cvtColor(image2, img_gray2, COLOR_BGR2GRAY);
    Mat thresh2;
    threshold(img_gray2, thresh2, 150, 255, THRESH_BINARY);

    vector<vector<Point>> contours3;
    vector<Vec4i> hierarchy3;
    findContours(thresh2, contours3, hierarchy3, RETR_LIST, CHAIN_APPROX_NONE);
    Mat image_copy4 = image2.clone();
    drawContours(image_copy4, contours3, -1, Scalar(0, 255, 0), 2);
    imshow("LIST", image_copy4);
    waitKey(0);
    imwrite("contours_retr_list.jpg", image_copy4);
    destroyAllWindows();

    vector<vector<Point>> contours4;
    vector<Vec4i> hierarchy4;
    findContours(thresh2, contours4, hierarchy4, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    Mat image_copy5 = image2.clone();
    drawContours(image_copy5, contours4, -1, Scalar(0, 255, 0), 2);
    imshow("EXTERNAL", image_copy5);
    waitKey(0);
    imwrite("contours_retr_external.jpg", image_copy4);
    destroyAllWindows();

    vector<vector<Point>> contours5;
    vector<Vec4i> hierarchy5;
    findContours(thresh2, contours5, hierarchy5, RETR_CCOMP, CHAIN_APPROX_NONE);
    Mat image_copy6 = image2.clone();
    drawContours(image_copy6, contours5, -1, Scalar(0, 255, 0), 2);
    imshow("EXTERNAL", image_copy6);
    waitKey(0);
    imwrite("contours_retr_ccomp.jpg", image_copy6);
    destroyAllWindows();

    vector<vector<Point>> contours6;
    vector<Vec4i> hierarchy6;
    findContours(thresh2, contours6, hierarchy6, RETR_TREE, CHAIN_APPROX_NONE);
    Mat image_copy7 = image2.clone();
    drawContours(image_copy7, contours6, -1, Scalar(0, 255, 0), 2);
    imshow("EXTERNAL", image_copy7);
    waitKey(0);
    imwrite("contours_retr_tree.jpg", image_copy7);
    destroyAllWindows();
}
