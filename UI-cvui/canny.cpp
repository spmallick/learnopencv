/*
This is a demo application to showcase the UI components of cvui.

Code licensed under the MIT license, check LICENSE file.
*/

#include <opencv2/opencv.hpp>
#include "cvui.h"

#define WINDOW_NAME	"CVUI Canny Edge"

int main(int argc, const char *argv[])
{
	cv::Mat lena = cv::imread("lena.jpg");
	cv::Mat frame = lena.clone();
	int low_threshold = 50, high_threshold = 150;
	bool use_canny = false;

	// Init a OpenCV window and tell cvui to use it.
	// If cv::namedWindow() is not used, mouse events will
	// not be captured by cvui.
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);

	while (true) {
		// Should we apply Canny edge?
		if (use_canny) {
			// Yes, we should apply it.
			cv::cvtColor(lena, frame, CV_BGR2GRAY);
			cv::Canny(frame, frame, low_threshold, high_threshold, 3);
		} else {
			// No, so just copy the original image to the displaying frame.
			lena.copyTo(frame);
		}

		// Render the settings window to house the checkbox
		// and the trackbars below.
		cvui::window(frame, 10, 50, 180, 180, "Settings");
		
		// Checkbox to enable/disable the use of Canny edge
		cvui::checkbox(frame, 15, 80, "Use Canny Edge", &use_canny);

		// Two trackbars to control the low and high threshold values
		// for the Canny edge algorithm.
		cvui::trackbar(frame, 15, 110, 165, &low_threshold, 5, 150);
		cvui::trackbar(frame, 15, 180, 165, &high_threshold, 80, 300);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();

		// Show everything on the screen
		cv::imshow(WINDOW_NAME, frame);

		// Check if ESC was pressed
		if (cv::waitKey(30) == 27) {
			break;
		}
	}

	return 0;
}