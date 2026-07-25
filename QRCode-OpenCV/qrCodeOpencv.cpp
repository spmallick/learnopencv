// Detect and decode one QR code with OpenCV 4.14 or OpenCV 5.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char* kExpectedData = "http://LearnOpenCV.com";

struct Options {
  fs::path input_path = fs::path(QR_PROJECT_DIR) / "qrcode-learnopencv.jpg";
  fs::path output_dir = "outputs";
  bool show_windows = true;
  bool validate = false;
};

struct DetectionResult {
  std::string data;
  std::vector<cv::Point2f> corners;
  cv::Mat rectified;
  double elapsed_seconds = 0.0;
};

void print_usage(const char* executable) {
  std::cout
      << "Usage: " << executable
      << " [--input IMAGE] [--output-dir DIR] [--no-display] [--validate]\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--input" && index + 1 < argc) {
      options.input_path = argv[++index];
    } else if (argument == "--output-dir" && index + 1 < argc) {
      options.output_dir = argv[++index];
    } else if (argument == "--no-display") {
      options.show_windows = false;
    } else if (argument == "--validate") {
      options.validate = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + argument);
    }
  }
  return options;
}

cv::Mat read_image(const fs::path& path) {
  const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Unable to read input image: " + path.string());
  }
  return image;
}

DetectionResult detect_qr_code(const cv::Mat& image) {
  cv::QRCodeDetector detector;
  cv::Mat points;
  cv::Mat straight_code;

  const auto started = std::chrono::steady_clock::now();
  const std::string data =
      detector.detectAndDecode(image, points, straight_code);
  const auto finished = std::chrono::steady_clock::now();

  std::vector<cv::Point2f> corners;
  if (!points.empty()) {
    const cv::Mat flattened = points.reshape(2, static_cast<int>(points.total()));
    for (int row = 0; row < flattened.rows; ++row) {
      corners.push_back(flattened.at<cv::Point2f>(row, 0));
    }
  }

  const double elapsed_seconds =
      std::chrono::duration<double>(finished - started).count();
  return {data, corners, straight_code, elapsed_seconds};
}

cv::Mat draw_corners(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& corners) {
  cv::Mat annotated = image.clone();
  if (corners.empty()) {
    return annotated;
  }

  std::vector<cv::Point> integer_corners;
  integer_corners.reserve(corners.size());
  for (const cv::Point2f& corner : corners) {
    integer_corners.emplace_back(cvRound(corner.x), cvRound(corner.y));
  }
  cv::polylines(
      annotated,
      integer_corners,
      true,
      cv::Scalar(255, 0, 0),
      3,
      cv::LINE_AA);
  return annotated;
}

void validate_result(const DetectionResult& result) {
  if (result.data != kExpectedData) {
    throw std::runtime_error(
        "Decoded payload mismatch: expected '" + std::string(kExpectedData) +
        "', got '" + result.data + "'");
  }
  if (result.corners.size() != 4U) {
    throw std::runtime_error(
        "Expected four QR-code corners, got " +
        std::to_string(result.corners.size()));
  }
  if (result.rectified.empty()) {
    throw std::runtime_error("OpenCV returned no rectified QR-code image");
  }
}

int run(const Options& options) {
  const cv::Mat image = read_image(options.input_path);
  const DetectionResult result = detect_qr_code(image);
  const cv::Mat annotated = draw_corners(image, result.corners);

  fs::create_directories(options.output_dir);
  const fs::path annotated_path = options.output_dir / "qr-code-annotated.png";
  if (!cv::imwrite(annotated_path.string(), annotated)) {
    throw std::runtime_error(
        "Unable to write annotated image: " + annotated_path.string());
  }

  if (!result.rectified.empty()) {
    const fs::path rectified_path = options.output_dir / "qr-code-rectified.png";
    if (!cv::imwrite(rectified_path.string(), result.rectified)) {
      throw std::runtime_error(
          "Unable to write rectified image: " + rectified_path.string());
    }
  }

  std::cout << "OpenCV version: " << CV_VERSION << '\n'
            << "Detect and decode time: " << result.elapsed_seconds
            << " seconds\n"
            << "Decoded data: "
            << (result.data.empty() ? "<none>" : result.data) << '\n'
            << "Annotated image: " << annotated_path << '\n';

  if (options.validate) {
    validate_result(result);
    std::cout << "VALIDATION PASSED: payload and four QR corners match\n";
  }

  if (options.show_windows) {
    cv::imshow("QR code result", annotated);
    if (!result.rectified.empty()) {
      cv::imshow("Rectified QR code", result.rectified);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
