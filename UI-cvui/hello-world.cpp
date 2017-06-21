/*
This is an extremely simple demo application to showcase the
principles of cvui.

Code licensed under the MIT license, check LICENSE file.
*/

#include <opencv2/opencv.hpp>
#include "cvui.h"

#define WINDOW_NAME "CVUI Hello World!"

int main(int argc, const char *argv[])
{
	cv::Mat frame = cv::Mat(200, 500, CV_8UC3);
	int count = 0;

	// Init a OpenCV window and tell cvui to use it.
	// If cv::namedWindow() is not used, mouse events will
	// not be captured by cvui.
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);

	while (true) {
		// Fill the frame with a nice color
		frame = cv::Scalar(49, 52, 49);

		// Buttons will return true if they were clicked, which makes
		// handling clicks a breeze.
		if (cvui::button(frame, 110, 80, "Hello, world!")) {
			// The button was clicked, so let's increment our counter.
			count++;
		}

		// Sometimes you want to show text that is not that simple, e.g. strings + numbers.
		// You can use cvui::printf for that. It accepts a variable number of parameter, pretty
		// much like printf does.
		// Let's show how many times the button has been clicked.
		cvui::printf(frame, 250, 90, 0.4, 0xff0000, "Button click count: %d", count);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();

		// Show everything on the screen
		cv::imshow(WINDOW_NAME, frame);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			break;
		}
	}

	return 0;
}