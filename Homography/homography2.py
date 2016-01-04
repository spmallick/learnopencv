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
    Mat im_src = imread(argv[1]);
    Size size = im_src.size();
   
    // Create a vector of points.
    vector<Point2f> pts_src;
    pts_src.push_back(Point2f(0,0));
    pts_src.push_back(Point2f(size.width - 1, 0));
    pts_src.push_back(Point2f(size.width - 1, size.height -1));
    pts_src.push_back(Point2f(0, size.height - 1 ));
    
    

    // Destination image
    Mat im_dst = imread(argv[2]);

    
    
    
    //Create a window
    namedWindow("Image", 1);
    
    Mat im_temp = im_dst.clone();
    
    userdata data;
    data.im = im_temp;


    
    //set the callback function for any mouse event
    setMouseCallback("Image", mouseHandler, &data);
    
    //show the image
    imshow("Image", im_temp);
    waitKey(0);
    
    Mat tform = findHomography(pts_src, data.points);
    warpPerspective(im_src, im_temp, tform, im_temp.size());
    
    Point pts_dst[4];
    for( int i = 0; i < 4; i++)
    {
        pts_dst[i] = data.points[i];
    }
    
    
    fillConvexPoly(im_dst, pts_dst, 4, Scalar(0), CV_AA);
    
    im_dst = im_dst + im_temp;
    
    imshow("Image", im_dst);
    waitKey(0);

    return 0;
}
