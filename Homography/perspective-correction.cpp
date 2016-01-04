#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct userdata{
    Mat im;
    vector<Point2f> points;
};


void mouseHandler(int event, int x, int y, int flags, void* data_ptr)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        userdata *data = ((userdata *) data_ptr);
        circle(data->im, Point(x,y),3,Scalar(0,0,255), 5, CV_AA);
        imshow("Image", data->im);
        if (data->points.size() < 4)
        {
            data->points.push_back(Point2f(x,y));
        }
    }
    
}



int main( int argc, char** argv)
{

    // Read source image.
    Mat im_src = imread("book1.jpg");

    // Destination image. The aspect ratio of the book is 3/4
    Size size(300,400);
    Mat im_dst = Mat::zeros(size,CV_8UC3);

    
    // Create a vector of destination points.
    vector<Point2f> pts_dst;
    
    pts_dst.push_back(Point2f(0,0));
    pts_dst.push_back(Point2f(size.width - 1, 0));
    pts_dst.push_back(Point2f(size.width - 1, size.height -1));
    pts_dst.push_back(Point2f(0, size.height - 1 ));
    
    // Set data for mouse event
    Mat im_temp = im_src.clone();
    userdata data;
    data.im = im_temp;

    cout << "Click on the four corners of the book -- top left first and" << endl
    << "bottom left last -- and then hit ENTER" << endl;
    
    // Show image and wait for 4 clicks. 
    imshow("Image", im_temp);
    // Set the callback function for any mouse event
    setMouseCallback("Image", mouseHandler, &data);
    waitKey(0);
    
    // Calculate the homography
    Mat h = findHomography(data.points, pts_dst);
    
    // Warp source image to destination
    warpPerspective(im_src, im_dst, h, size);
    
    // Show image
    imshow("Image", im_dst);
    waitKey(0);

    return 0;
}
