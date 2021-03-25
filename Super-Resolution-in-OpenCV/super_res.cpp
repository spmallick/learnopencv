#include <opencv2/dnn_superres.hpp>

img=cv::imread("image.png")
DnnSuperResImpl sr;
string model_path = "ESPCN_x4.pb";
sr.readModel(model_path);
sr.setModel("espcn", 4); // set the model by passing the value and the upsampling ratio
Mat result; // creating blank mat for result
sr.upsample(img, result); // upscale the input image
cv::imwrite("output.png",result)
