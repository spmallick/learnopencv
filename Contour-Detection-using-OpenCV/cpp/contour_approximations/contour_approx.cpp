#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // read the image
    Mat image = imread("../../input/image_1.jpg");
    // convert the image to grayscale format
    Mat img_gray;
    cvtColor(image, img_gray, COLOR_BGR2GRAY);
    // apply binary thresholding
    Mat thresh;
    threshold(img_gray, thresh, 150, 255, THRESH_BINARY);
    imshow("Binary mage", thresh);
    waitKey(0);
    imwrite("image_thres1.jpg", thresh);
    destroyAllWindows();

    // detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    // draw contours on the original image
    Mat image_copy = image.clone();
    drawContours(image_copy, contours, -1, Scalar(0, 255, 0), 2);
    imshow("None approximation", image_copy);
    waitKey(0);
    imwrite("contours_none_image1.jpg", image_copy);
    destroyAllWindows();


    // Now let's try with CHAIN_APPROX_SIMPLE`
    // detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    vector<vector<Point>> contours1;
    vector<Vec4i> hierarchy1;
    findContours(thresh, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_SIMPLE);
    // draw contours on the original image
    Mat image_copy1 = image.clone();
    drawContours(image_copy1, contours1, -1, Scalar(0, 255, 0), 2);
    imshow("Simple approximation", image_copy1);
    waitKey(0);
    imwrite("contours_simple_image1.jpg", image_copy1);
    destroyAllWindows();

    // using a proper image for visualizing CHAIN_APPROX_SIMPLE
    Mat image1 = imread("../../input/image_2.jpg");
    Mat img_gray1;
    cvtColor(image1, img_gray1, COLOR_BGR2GRAY);
    Mat thresh1;
    threshold(img_gray1, thresh1, 150, 255, THRESH_BINARY);
    vector<vector<Point>> contours2;
    vector<Vec4i> hierarchy2;
    findContours(thresh1, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_NONE);
    Mat image_copy2 = image1.clone();
    drawContours(image_copy2, contours2, -1, Scalar(0, 255, 0), 2);
    imshow("None approximation", image_copy2);
    waitKey(0);
    imwrite("contours_none_image1.jpg", image_copy2);
    destroyAllWindows();
    Mat image_copy3 = image1.clone();
    for(int i=0; i<contours2.size(); i=i+1){
        for (int j=0; j<contours2[i].size(); j=j+1){
            circle(image_copy3, (contours2[i][0], contours2[i][1]), 2, 
            Scalar(0, 255, 0), 2);
        }
    }
    imshow("CHAIN_APPROX_SIMPLE Point only", image_copy3);
    waitKey(0);
    imwrite("contour_point_simple.jpg", image_copy3);
    destroyAllWindows();
}
