#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef HU_MOMENTS_IMAGE_DIR
#define HU_MOMENTS_IMAGE_DIR "images"
#endif

namespace {

cv::Mat readBinaryImage(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    throw std::runtime_error("Could not read image: " + path.string());
  }
  cv::threshold(image, image, 128, 255, cv::THRESH_BINARY);
  return image;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 1 && argc != 4) {
    std::cerr
        << "Usage: shapeMatcher [REFERENCE DIFFERENT TRANSFORMED]\n";
    return 2;
  }

  const std::filesystem::path imageDir = HU_MOMENTS_IMAGE_DIR;
  const std::filesystem::path referencePath =
      argc == 4 ? argv[1] : imageDir / "S0.png";
  const std::filesystem::path differentPath =
      argc == 4 ? argv[2] : imageDir / "K0.png";
  const std::filesystem::path transformedPath =
      argc == 4 ? argv[3] : imageDir / "S4.png";

  try {
    const cv::Mat reference = readBinaryImage(referencePath);
    const cv::Mat different = readBinaryImage(differentPath);
    const cv::Mat transformed = readBinaryImage(transformedPath);

    const double selfDistance = cv::matchShapes(
        reference, reference, cv::CONTOURS_MATCH_I2, 0.0);
    const double differentDistance = cv::matchShapes(
        reference, different, cv::CONTOURS_MATCH_I2, 0.0);
    const double transformedDistance = cv::matchShapes(
        reference, transformed, cv::CONTOURS_MATCH_I2, 0.0);

    std::cout << "Shape distances\n"
              << "---------------\n"
              << std::fixed << std::setprecision(12)
              << referencePath.filename().string() << " and "
              << referencePath.filename().string() << ": " << selfDistance
              << '\n'
              << referencePath.filename().string() << " and "
              << differentPath.filename().string() << ": "
              << differentDistance << '\n'
              << referencePath.filename().string() << " and "
              << transformedPath.filename().string() << ": "
              << transformedDistance << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
