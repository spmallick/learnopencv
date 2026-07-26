#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../homography_utils.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testCornerMapping() {
    const auto source = homography_example::rectangleCorners(120, 80);
    const auto destination =
        homography_example::parsePoints("20,30;180,20;190,140;10,150");
    const cv::Mat homography =
        homography_example::computeHomography(source, destination);

    std::vector<cv::Point2f> transformed;
    cv::perspectiveTransform(source, transformed, homography);
    for (std::size_t index = 0; index < destination.size(); ++index) {
        require(
            cv::norm(transformed[index] - destination[index]) < 1e-3,
            "Homography did not map a corner accurately.");
    }
}

void testRectification() {
    cv::Mat image = cv::Mat::zeros(100, 120, CV_8UC3);
    cv::rectangle(
        image, {20, 20}, {99, 79}, cv::Scalar(0, 255, 0), cv::FILLED);
    const auto points =
        homography_example::parsePoints("20,20;99,20;99,79;20,79");

    const auto [rectified, homography] =
        homography_example::rectifyImage(image, points, 80, 60);
    static_cast<void>(homography);
    require(
        rectified.size() == cv::Size(80, 60),
        "Rectification returned the wrong dimensions.");
    require(
        rectified.at<cv::Vec3b>(30, 40)[1] > 240,
        "Rectification did not preserve the source content.");
}

void testCompositeMask() {
    const cv::Mat source(
        40, 60, CV_8UC3, cv::Scalar(0, 0, 255));
    const cv::Mat destination(
        120, 180, CV_8UC3, cv::Scalar(255, 0, 0));
    const auto points =
        homography_example::parsePoints("40,30;139,30;139,89;40,89");

    const auto [result, homography] =
        homography_example::compositeOnQuad(
            source, destination, points);
    static_cast<void>(homography);
    require(
        result.at<cv::Vec3b>(5, 5) == destination.at<cv::Vec3b>(5, 5),
        "Composite changed pixels outside the destination quad.");
    const auto center = result.at<cv::Vec3b>(60, 90);
    require(
        center[2] > 240 && center[0] < 10,
        "Composite did not copy source pixels into the quad.");
}

void testInvalidQuad() {
    bool rejected = false;
    try {
        static_cast<void>(
            homography_example::parsePoints("0,0;10,10;0,10;10,0"));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "A self-intersecting quad was not rejected.");
}

}  // namespace

int main() {
    try {
        testCornerMapping();
        testRectification();
        testCompositeMask();
        testInvalidQuad();
        std::cout << "All homography C++ tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
