#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void sepia(Mat img){
	Mat res = img.clone();
	cvtColor(res,res,COLOR_BGR2RGB);
	transform(res,res,Matx33f(0.393,0.769,0.189,
				0.349,0.686,0.168,
				0.272,0.534,0.131));
	cvtColor(res,res,COLOR_RGB2BGR);
	imshow("original",img);
	imshow("Sepia",res);
	waitKey(0);
	destroyAllWindows();
}

int main(){
	Mat img = imread("image.jpg");
	sepia(img);
	return 0;
}
