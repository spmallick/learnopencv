#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


void nothing(int x, void* data) {}

void tv_60(Mat img) {
	
	namedWindow("image");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",&slider,255,nothing);
	createTrackbar("threshold","image",&slider2,100,nothing);

	while (true) {
		int height = img.size().height;
		int width = img.size().width;
		Mat gray;
		cvtColor(img, gray, COLOR_BGR2GRAY);
		float thresh = getTrackbarPos("threshold","image");
		float val = getTrackbarPos("val","image");

		for (int i=0; i < height; i++){
			for (int j=0; j < width; j++){
				if (rand()%100 <= thresh){
					if (rand()%2 == 0)
						gray.at<uchar>(i,j) = std::min(gray.at<uchar>(i,j) + rand()%((int)val+1), 255);
					else
						gray.at<uchar>(i,j) = std::max(gray.at<uchar>(i,j) - rand()%((int)val+1), 0);
				}
			}
		}

    		imshow("original",img);
    		imshow("image",gray);

	    	if (waitKey(1) == 'q')
	    		break;
		}
	destroyAllWindows();
}

int main(){
	Mat img = imread("image.jpg");
	tv_60(img);
	return 0;
}
