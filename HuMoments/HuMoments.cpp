#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

double logTransform(double value) {
  if (value == 0.0) {
    return 0.0;
  }
  return -std::copysign(1.0, value) * std::log10(std::abs(value));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: HuMoments IMAGE [IMAGE ...]\n";
    return 2;
  }

  for (int argument = 1; argument < argc; ++argument) {
    const std::string filename = argv[argument];
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
      std::cerr << "Could not read image: " << filename << '\n';
      return 1;
    }

    cv::threshold(image, image, 128, 255, cv::THRESH_BINARY);
    const cv::Moments moments = cv::moments(image);

    double huMoments[7];
    cv::HuMoments(moments, huMoments);

    std::cout << filename << ": " << std::fixed << std::setprecision(5);
    for (const double moment : huMoments) {
      std::cout << logTransform(moment) << ' ';
    }
    std::cout << '\n';
  }

  return 0;
}
