#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


void nothing(int x, void* data) {}

Mat kernel_generator(int size){
	Mat kernel = Mat(size,size,CV_8S,Scalar(0));
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			if (i < j){
				kernel.at<schar>(i,j) = -1;
			}
			else if (i > j){
				kernel.at<schar>(i,j) = 1;
			}
		}
	}
	return kernel;
}

void emboss(Mat img){
	namedWindow("image");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("size","image",&slider,8,nothing);
	createTrackbar("0 : BL n1 : BR n2 : TR n3 : BR","image",&slider2,3,nothing);

	while (true){
		int size = getTrackbarPos("size","image");
		size += 2;
		int s = getTrackbarPos("0 : BL n1 : BR n2 : TR n3 : BR","image");
		int height = img.size().height;
		int width = img.size().width;
		Mat y = Mat(height,width,CV_8U,Scalar(128));
		Mat gray;
		cvtColor(img,gray,COLOR_BGR2GRAY);
		Mat kernel = kernel_generator(size);

		for (int i=0; i<s; i++)
			rotate(kernel,kernel,ROTATE_90_COUNTERCLOCKWISE);

		Mat dst;
		filter2D(gray,dst,-1,kernel);
		Mat res;
		add(dst,y,res);

		imshow("Original",img);
		imshow("image",res);

		if (waitKey(1) == 'q')
                        break;
                }
        destroyAllWindows();
}

int main(){
	Mat img = imread("image.jpg");
	emboss(img);
	return 0;
}
