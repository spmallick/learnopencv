#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


void nothing(int x, void* data) {}

void brightness(Mat img) {
	

	namedWindow("image");
	int slider = 100;
	createTrackbar("val","image",&slider,150,nothing);
	Mat hsv;

	while (true) {
		cvtColor(img, hsv, COLOR_BGR2HSV);
		float val = getTrackbarPos("val","image");
		val=val/100.0;
		Mat channels[3];
		split(hsv,channels);
		Mat H = channels[0];
		H.convertTo(H,CV_32F);
		Mat S = channels[1];
		S.convertTo(S,CV_32F);
		Mat V = channels[2];
		V.convertTo(V,CV_32F);

		for (int i=0; i < H.size().height; i++){
			for (int j=0; j < H.size().width; j++){
	// scale pixel values up or down for channel 1(Saturation)
				S.at<float>(i,j) *= val;
				if (S.at<float>(i,j) > 255)
					S.at<float>(i,j) = 255;

	// scale pixel values up or down for channel 2(Value)
				V.at<float>(i,j) *= val;
				if (V.at<float>(i,j) > 255)
					V.at<float>(i,j) = 255;
			}
		}
		H.convertTo(H,CV_8U);
		S.convertTo(S,CV_8U);
		V.convertTo(V,CV_8U);

		vector<Mat> hsvChannels{H,S,V};
		Mat hsvNew;
    		merge(hsvChannels,hsvNew);

    		Mat res;
    		cvtColor(hsvNew,res,COLOR_HSV2BGR);

    		imshow("original",img);
    		imshow("image",res);

	    	if (waitKey(1) == 'q')
	    		break;
		}
	destroyAllWindows();
}

void tv_60(Mat img) {
	
	namedWindow("image");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",&slider,255,nothing);
	createTrackbar("threshold","image",&slider,100,nothing);

	while (true) {
		int width = img.size().height;
		int height = img.size().width;
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

Mat exponential_function(Mat channel, float exp){
	Mat table(1, 256, CV_8U);

	for (int i = 0; i < 256; i++)
		table.at<uchar>(i) = min((int)pow(i,exp),255);

	LUT(channel,table,channel);
	return channel;
}

void duo_tone(Mat img){
	namedWindow("image");
	int slider1 = 0;
	int slider2 = 1;
	int slider3 = 3;
	int slider4 = 0;
	string switch1 = "0 : BLUE n1 : GREEN n2 : RED";
	string switch2 = "0 : BLUE n1 : GREEN n2 : RED n3 : NONE";
	string switch3 = "0 : DARK n1 : LIGHT";
	createTrackbar("exponent","image",&slider1,10,nothing);
	createTrackbar(switch1,"image",&slider2,2,nothing);
	createTrackbar(switch2,"image",&slider3,3,nothing);
	createTrackbar(switch3,"image",&slider4,1,nothing);

	while(true){
		int exp1 = getTrackbarPos("exponent","image");
		float exp = 1 + exp1/100.0;
		int s1 = getTrackbarPos(switch1,"image");
		int s2 = getTrackbarPos(switch2,"image");
		int s3 = getTrackbarPos(switch3,"image");
		Mat res = img.clone();
		Mat channels[3];
		split(img,channels);
		for (int i=0; i<3; i++){
			if ((i == s1)||(i==s2)){
				channels[i] = exponential_function(channels[i],exp);
			}
			else{
				if (s3){
					channels[i] = exponential_function(channels[i],2-exp);
				}
				else{
					channels[i] = Mat::zeros(channels[i].size(),CV_8UC1);
				}
			}
		}
		vector<Mat> newChannels{channels[0],channels[1],channels[2]};
		merge(newChannels,res);
		imshow("Original",img);
		imshow("image",res);
		if (waitKey(1) == 'q')
                        break;
                }
        destroyAllWindows();
}

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
	duo_tone(img);
	return 0;
}
