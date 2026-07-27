#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "otsu_threshold.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compareWithOpenCv(const cv::Mat& image) {
    cv::Mat expectedBinary;
    const int expectedThreshold = static_cast<int>(cv::threshold(
        image, expectedBinary, 0.0, 255.0,
        cv::THRESH_BINARY | cv::THRESH_OTSU));
    const int actualThreshold = learnopencv::otsu::threshold(image);
    const cv::Mat actualBinary =
        learnopencv::otsu::apply(image, actualThreshold);

    require(actualThreshold == expectedThreshold, "Threshold mismatch.");
    require(cv::countNonZero(actualBinary != expectedBinary) == 0,
            "Binary output mismatch.");
}

}  // namespace

int main() {
    try {
        const auto boatPath =
            std::string(OTSU_SOURCE_DIR) + "/boat.jpg";
        const cv::Mat boat =
            cv::imread(boatPath, cv::IMREAD_GRAYSCALE);
        require(!boat.empty(), "Could not read boat.jpg.");
        require(learnopencv::otsu::threshold(boat) == 132,
                "boat.jpg threshold must be 132.");
        compareWithOpenCv(boat);

        cv::Mat bimodal(64, 64, CV_8UC1, cv::Scalar(20));
        bimodal.colRange(32, 64).setTo(200);
        compareWithOpenCv(bimodal);

        for (const int value : {0, 128, 255}) {
            compareWithOpenCv(cv::Mat(16, 16, CV_8UC1, cv::Scalar(value)));
        }

        bool emptyRejected = false;
        try {
            static_cast<void>(learnopencv::otsu::threshold(cv::Mat{}));
        } catch (const std::invalid_argument&) {
            emptyRejected = true;
        }
        require(emptyRejected, "An empty image must be rejected.");

        bool colorRejected = false;
        try {
            static_cast<void>(learnopencv::otsu::threshold(
                cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(0))));
        } catch (const std::invalid_argument&) {
            colorRejected = true;
        }
        require(colorRejected, "A color image must be rejected.");

        std::cout << "All Otsu C++ regression checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Otsu regression failure: " << error.what() << '\n';
        return 1;
    }
}
