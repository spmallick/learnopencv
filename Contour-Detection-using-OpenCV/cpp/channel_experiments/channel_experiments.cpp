#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // read the image
    Mat image = imread("../../input/image_1.jpg");

    // B, G, R channel splitting
    Mat channels[3];
    split(image, channels);

    // detect contours using blue channel and without thresholding
    vector<vector<Point>> contours1;
    vector<Vec4i> hierarchy1;
    findContours(channels[0], contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_NONE);
    // draw contours on the original image
    Mat image_contour_blue = image.clone();
    drawContours(image_contour_blue, contours1, -1, Scalar(0, 255, 0), 2);
    imshow("Contour detection using blue channels only", image_contour_blue);
    waitKey(0);
    imwrite("blue_channel.jpg", image_contour_blue);
    destroyAllWindows();

    // detect contours using green channel and without thresholding
    vector<vector<Point>> contours2;
    vector<Vec4i> hierarchy2;
    findContours(channels[1], contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_NONE);
    // draw contours on the original image
    Mat image_contour_green = image.clone();
    drawContours(image_contour_green, contours2, -1, Scalar(0, 255, 0), 2);
    imshow("Contour detection using green channels only", image_contour_green);
    waitKey(0);
    imwrite("green_channel.jpg", image_contour_green);
    destroyAllWindows();

    // detect contours using red channel and without thresholding
    vector<vector<Point>> contours3;
    vector<Vec4i> hierarchy3;
    findContours(channels[2], contours3, hierarchy3, RETR_TREE, CHAIN_APPROX_NONE);
    // draw contours on the original image
    Mat image_contour_red = image.clone();
    drawContours(image_contour_red, contours3, -1, Scalar(0, 255, 0), 2);
    imshow("Contour detection using red channels only", image_contour_red);
    waitKey(0);
    imwrite("red_channel.jpg", image_contour_red);
    destroyAllWindows();
}
