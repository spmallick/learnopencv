#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


void nothing(int x, void* data) {}

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

int main(){
	Mat img = imread("image.jpg");
	duo_tone(img);
	return 0;
}
