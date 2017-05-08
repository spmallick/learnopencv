#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//Global Variables
Mat img, placeholder;

// Callback function for any event on he mouse
void onMouse( int event, int x, int y, int flags, void* userdata )
{   
    if( event == EVENT_MOUSEMOVE )
	{

     	Vec3b bgrPixel(img.at<Vec3b>(y, x));
        
        Mat3b hsv,ycb,lab;
        // Create Mat object from vector since cvtColor accepts a Mat object
        Mat3b bgr (bgrPixel);
        
        //Convert the single pixel BGR Mat to other formats
        cvtColor(bgr, ycb, COLOR_BGR2YCrCb);
        cvtColor(bgr, hsv, COLOR_BGR2HSV);
        cvtColor(bgr, lab, COLOR_BGR2Lab);
        
        //Get back the vector from Mat
        Vec3b hsvPixel(hsv.at<Vec3b>(0,0));
        Vec3b ycbPixel(ycb.at<Vec3b>(0,0));
        Vec3b labPixel(lab.at<Vec3b>(0,0));
       
        // Create an empty placeholder for displaying the values
        placeholder = Mat::zeros(img.rows,400,CV_8UC3);

        //fill the placeholder with the values of color spaces
        putText(placeholder, format("BGR [%d, %d, %d]",bgrPixel[0],bgrPixel[1],bgrPixel[2]), Point(20, 70), FONT_HERSHEY_COMPLEX, .9, Scalar(255,255,255), 1);
        putText(placeholder, format("HSV [%d, %d, %d]",hsvPixel[0],hsvPixel[1],hsvPixel[2]), Point(20, 140), FONT_HERSHEY_COMPLEX, .9, Scalar(255,255,255), 1);
        putText(placeholder, format("YCrCb [%d, %d, %d]",ycbPixel[0],ycbPixel[1],ycbPixel[2]), Point(20, 210), FONT_HERSHEY_COMPLEX, .9, Scalar(255,255,255), 1);
        putText(placeholder, format("LAB [%d, %d, %d]",labPixel[0],labPixel[1],labPixel[2]), Point(20, 280), FONT_HERSHEY_COMPLEX, .9, Scalar(255,255,255), 1);


	    Size sz1 = img.size();
	    Size sz2 = placeholder.size();
	    
        //Combine the two results to show side by side in a single image
        Mat combinedResult(sz1.height, sz1.width+sz2.width, CV_8UC3);
	    Mat left(combinedResult, Rect(0, 0, sz1.width, sz1.height));
	    img.copyTo(left);
	    Mat right(combinedResult, Rect(sz1.width, 0, sz2.width, sz2.height));
	    placeholder.copyTo(right);
	    imshow("PRESS P for Previous, N for Next Image", combinedResult);
    }
}

int main( int argc, const char** argv )
{
    // filename
    // Read the input image
    int image_number = 0;
    int nImages = 10;

    if(argc > 1)
        nImages = atoi(argv[1]);
    
    char filename[20];
    sprintf(filename,"images/rub%02d.jpg",image_number%nImages);
    img = imread(filename);
    // Resize the image to 400x400
    Size rsize(400,400);
    resize(img,img,rsize);

    if(img.empty())
    {
        return -1;
    }
    
    // Create an empty window
    namedWindow("PRESS P for Previous, N for Next Image", WINDOW_AUTOSIZE);   
    // Create a callback function for any event on the mouse
    setMouseCallback( "PRESS P for Previous, N for Next Image", onMouse );
    
    imshow( "PRESS P for Previous, N for Next Image", img );
    while(1)
    {
        char k = waitKey(1) & 0xFF;
        if (k == 27)
            break;
        //Check next image in the folder
        if (k =='n')
        {
            image_number++;
            sprintf(filename,"images/rub%02d.jpg",image_number%nImages);
            img = imread(filename);
            resize(img,img,rsize); 
        }
        //Check previous image in he folder
        else if (k =='p')
        {
            image_number--;
            sprintf(filename,"images/rub%02d.jpg",image_number%nImages);
            img = imread(filename);
            resize(img,img,rsize);
        }
    }
    return 0;
}