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
        circle(data->im, Point(x,y),3,Scalar(0,255,255), 5, CV_AA);
        imshow("Image", data->im);
        if (data->points.size() < 4)
        {
            data->points.push_back(Point2f(x,y));
        }
    }
    
}



int main( int argc, char** argv)
{

    // Read in the image.
    Mat im_src = imread("book1.jpg");

    // Destination image
    Size size(300,400);
    Mat im_dst = Mat::zeros(size,CV_8UC3);

    
    // Create a vector of points.
    vector<Point2f> pts_dst;
    
    
    
    pts_dst.push_back(Point2f(0,0));
    pts_dst.push_back(Point2f(size.width - 1, 0));
    pts_dst.push_back(Point2f(size.width - 1, size.height -1));
    pts_dst.push_back(Point2f(0, size.height - 1 ));
    
    
    
    //Create a window
    namedWindow("Image", 1);
    
    Mat im_temp = im_src.clone();
    
    userdata data;
    data.im = im_temp;


    
    //set the callback function for any mouse event
    setMouseCallback("Image", mouseHandler, &data);
    
    //show the image
    imshow("Image", im_temp);
    waitKey(0);
    
    Mat tform = findHomography(data.points, pts_dst);
    warpPerspective(im_src, im_dst, tform, size);
    
    imshow("Image", im_dst);
    waitKey(0);

    return 0;
}
