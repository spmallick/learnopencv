#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

int main(){

	// Read image
	cv::Mat img = cv::imread("image.png");

	// Make DNN Super resolution instance
	cv::dnn_superres::DnnSuperResImpl sr;

	// Read the model
	std::string model_path = "ESPCN_x4.pb";
	sr.readModel(model_path);

	// Set the model by passing the value and the upsampling ratio
	sr.setModel("espcn", 4);

	// Creating a blank Mat for result
	cv::Mat result;

	// Upscale the input image
	sr.upsample(img, result);

	// Write the final image
	cv::imwrite("output.png",result);

  return 0;
  
}
