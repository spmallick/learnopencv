#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;

int computeMedian(vector<int> elements) {
	nth_element(elements.begin(), elements.begin()+elements.size()/2, elements.end());
	//sort(elements.begin(),elements.end());

	return elements[elements.size()/2];
}

cv::Mat compute_median(std::vector<cv::Mat> vec) {
	// Note: Expects the image to be CV_8UC3
	cv::Mat medianImg(vec[0].rows, vec[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));

	for(int row=0; row<vec[0].rows; row++) {
		for(int col=0; col<vec[0].cols; col++) {
			std::vector<int> elements_B;
			std::vector<int> elements_G;
			std::vector<int> elements_R;

			for(int imgNumber=0; imgNumber<vec.size(); imgNumber++) {	
				int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
				int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
				int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];
				
				elements_B.push_back(B);
				elements_G.push_back(G);
				elements_R.push_back(R);
			}

			medianImg.at<cv::Vec3b>(row, col)[0] = computeMedian(elements_B);
			medianImg.at<cv::Vec3b>(row, col)[1] = computeMedian(elements_G);
			medianImg.at<cv::Vec3b>(row, col)[2] = computeMedian(elements_R);
		}
	}
	return medianImg;
}

int main(int argc, char const *argv[])
{	
	std::string video_file;
	// Read video file
	if(argc > 1) {
		video_file = argv[1];
	} else {
		video_file = "video.mp4";
	}

	VideoCapture cap(video_file);

	if(!cap.isOpened())
		cerr << "Error opening video file\n";

	// Randomly select 25 frames
	default_random_engine generator;
	uniform_int_distribution<int>distribution(0, cap.get(CAP_PROP_FRAME_COUNT));

	vector<Mat> frames;
	Mat frame;

	for(int i=0; i<25; i++) {
		int fid = distribution(generator);
		cap.set(CAP_PROP_POS_FRAMES, fid);
		Mat frame;
		cap >> frame;
		if(frame.empty())
			continue;
		frames.push_back(frame);
	}

	// Calculate the median along the time axis
	Mat medianFrame = compute_median(frames);

	// Display median frame
	imshow("frame", medianFrame);
	waitKey(0);

	//  Reset frame number to 0
	cap.set(CAP_PROP_POS_FRAMES, 0);

	// Convert background to grayscale
	Mat grayMedianFrame;
	cvtColor(medianFrame, grayMedianFrame, COLOR_BGR2GRAY);

	// Loop over all frames
	while(1) {
		// Read frame
		cap >> frame;

		if (frame.empty())
			break;

		// Convert current frame to grayscale
		cvtColor(frame, frame, COLOR_BGR2GRAY);

		// Calculate absolute difference of current frame and the median frame
		Mat dframe;
		absdiff(frame, grayMedianFrame, dframe);

		// Threshold to binarize
		threshold(dframe, dframe, 30, 255, THRESH_BINARY);
		
		// Display Image
		imshow("frame", dframe);
		waitKey(20);
	}

	cap.release();
	return 0;
}
