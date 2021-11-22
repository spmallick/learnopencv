// Include Libraries
#include <chrono>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/wechat_qrcode.hpp>


using namespace std;
using namespace cv;
using namespace std::chrono;

// Utility function to draw boundig box.
void display(Mat &im, Mat &bbox)
{
	int n = bbox.rows;
	for (int i = 0; i < n; i++)
	{
		line(im, Point2i(bbox.at<float>(i,0),bbox.at<float>(i,1)), 
		Point2i(bbox.at<float>((i+1) % n,0), bbox.at<float>((i+1) % n,1)), 
		Scalar(0,255,0), 3);
	}
	imshow("Image", im);
}


int main()
{
	// Instantiate WeChat QR Code detector.
	Ptr<wechat_qrcode::WeChatQRCode> detector;
	detector = makePtr<wechat_qrcode::WeChatQRCode>("../model/detect.prototxt", 
	"../model/detect.caffemodel",
	"../model/sr.prototxt", 
	"../model/sr.caffemodel");

	// Read image.
	Mat img;
	img = imread("sample-qrcode.jpg");

	vector<Mat> points;
	// Start time.
	auto start = high_resolution_clock::now();
	// Detect and decode.
	auto res = detector->detectAndDecode(img, points);
	// End time.
	auto stop = high_resolution_clock::now();
	// Time taken in milliseconds.
	auto duration = duration_cast<milliseconds>(stop - start);

	if (res.size() > 0)
	{
		// Print detected data.
		for (const auto& value : res) 
		{
			cout << value << endl;
		}
		// Convert to Mat.
		Mat1f matBbox;
		for(int i=0; i<points[0].size().height; i++)
		{
			matBbox.push_back( points[0].row(i));
		}
		cout << "Time taken : " << duration.count() << " milliseconds" << endl;
		cout << matBbox << endl;
		// Display bounding box. 
		display(img, matBbox);
	}
	else
	cout << "QR Code not detected." << endl;
	imshow("Image", img);
	waitKey(0);
	return 0;
}
