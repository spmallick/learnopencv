#include <opencv2/core/utility.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const char* kKeys =
    "{help h usage ? |      | Show this message }"
    "{input i        |sample.jpg| Input image }"
    "{output o       |convex-hull-output.jpg| Output visualization }"
    "{threshold t    |200   | Binary threshold from 0 to 255 }"
    "{display        |false | Show GUI windows }";

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, kKeys);
    parser.about("OpenCV convex-hull example");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    const std::string image_path = parser.get<std::string>("input");
    const fs::path output_path = parser.get<std::string>("output");
    const int threshold_value = parser.get<int>("threshold");
    const bool display = parser.get<bool>("display");
    if (threshold_value < 0 || threshold_value > 255) {
        std::cerr << "Error: --threshold must be between 0 and 255.\n";
        return 2;
    }

    const cv::Mat source = cv::imread(image_path, cv::IMREAD_COLOR);
    if (source.empty()) {
        std::cerr << "Error: input image not found or unreadable: "
                  << image_path << '\n';
        return 2;
    }

    cv::Mat gray;
    cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::blur(gray, blurred, cv::Size(3, 3));
    cv::Mat binary;
    cv::threshold(
        blurred, binary, threshold_value, 255, cv::THRESH_BINARY
    );

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
        binary,
        contours,
        hierarchy,
        cv::RETR_TREE,
        cv::CHAIN_APPROX_SIMPLE
    );

    std::vector<std::vector<cv::Point>> hulls(contours.size());
    for (std::size_t index = 0; index < contours.size(); ++index) {
        cv::convexHull(contours[index], hulls[index], false);
    }

    cv::Mat drawing = cv::Mat::zeros(binary.size(), CV_8UC3);
    for (std::size_t index = 0; index < contours.size(); ++index) {
        const int contour_index = static_cast<int>(index);
        cv::drawContours(
            drawing,
            contours,
            contour_index,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_8,
            hierarchy
        );
        cv::drawContours(
            drawing,
            hulls,
            contour_index,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_8
        );
    }

    try {
        if (output_path.has_parent_path()) {
            fs::create_directories(output_path.parent_path());
        }
    } catch (const fs::filesystem_error& error) {
        std::cerr << "Filesystem error: " << error.what() << '\n';
        return 3;
    }
    if (!cv::imwrite(output_path.string(), drawing)) {
        std::cerr << "Error: OpenCV could not write: " << output_path << '\n';
        return 3;
    }
    std::cout << "Detected " << contours.size() << " contours and wrote "
              << output_path << '\n';

    if (display) {
        cv::imshow("Source", source);
        cv::imshow("Contours and convex hulls", drawing);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
