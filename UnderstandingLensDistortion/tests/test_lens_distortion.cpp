#include "lens_calibration.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef LENS_SOURCE_DIR
#define LENS_SOURCE_DIR "."
#endif

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string& label) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            label + " expected " + std::to_string(expected) +
            " +/- " + std::to_string(tolerance) + ", got " +
            std::to_string(actual));
    }
}

template <typename Callable>
void requireThrows(Callable&& callable, const std::string& label) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label + " did not throw.");
}

}  // namespace

int main() {
    try {
        const auto projectDirectory =
            std::filesystem::path(LENS_SOURCE_DIR);
        const auto paths = lens_distortion::discoverImages(
            projectDirectory / "images");
        require(paths.size() == 41U, "Expected all 41 calibration images.");

        const auto calibration =
            lens_distortion::calibrateFromImages(
                paths, cv::Size(6, 9), 1.0, true);
        require(
            calibration.successfulImages.size() == 41U,
            "Expected checkerboard detection in all 41 images.");
        require(
            calibration.failedImages.empty(),
            "No checkerboard detection should fail.");
        require(
            calibration.imageSize == cv::Size(640, 480),
            "Calibration must preserve width=640 and height=480.");
        requireNear(calibration.rms, 0.26032, 0.02, "Calibration RMS");
        requireNear(
            calibration.reprojectionRMSE,
            calibration.rms,
            1e-5,
            "Reprojection RMSE");
        requireNear(
            calibration.cameraMatrix.at<double>(0, 0),
            503.51,
            3.0,
            "fx");
        requireNear(
            calibration.cameraMatrix.at<double>(1, 1),
            503.15,
            3.0,
            "fy");
        requireNear(
            calibration.cameraMatrix.at<double>(0, 2),
            313.41,
            3.0,
            "cx");
        requireNear(
            calibration.cameraMatrix.at<double>(1, 2),
            243.09,
            3.0,
            "cy");

        const cv::Mat sample = cv::imread(
            calibration.successfulImages.front().string(),
            cv::IMREAD_COLOR);
        require(!sample.empty(), "Could not read test image.");
        const auto direct = lens_distortion::undistortImage(
            sample, calibration, 1.0, "direct", false);
        const auto remapped = lens_distortion::undistortImage(
            sample, calibration, 1.0, "remap", false);
        require(
            direct.image.size() == sample.size(),
            "Undistortion should preserve image dimensions.");
        requireNear(direct.roi.x, 5.0, 2.0, "ROI x");
        requireNear(direct.roi.y, 7.0, 2.0, "ROI y");
        requireNear(direct.roi.width, 627.0, 2.0, "ROI width");
        requireNear(direct.roi.height, 466.0, 2.0, "ROI height");
        require(
            direct.roi == remapped.roi,
            "Direct and remap paths must share the ROI.");

        cv::Mat difference;
        cv::absdiff(direct.image, remapped.image, difference);
        double maximumDifference = 0.0;
        cv::minMaxLoc(difference.reshape(1), nullptr, &maximumDifference);
        const cv::Scalar meanDifference = cv::mean(difference);
        const double meanAcrossChannels =
            (meanDifference[0] + meanDifference[1] +
             meanDifference[2]) /
            3.0;
        require(
            maximumDifference <= 8.0,
            "Direct/remap maximum difference exceeds tolerance: " +
                std::to_string(maximumDifference));
        require(
            meanAcrossChannels < 0.1,
            "Direct/remap mean difference exceeds tolerance: " +
                std::to_string(meanAcrossChannels));

        const auto cropped = lens_distortion::undistortImage(
            sample, calibration, 1.0, "remap", true);
        require(
            cropped.image.size() == cropped.roi.size(),
            "Cropped output must match the valid-pixel ROI.");

        requireThrows(
            [] {
                lens_distortion::calibrateFromImages({});
            },
            "Empty image list");
        requireThrows(
            [&] {
                lens_distortion::undistortImage(
                    cv::Mat(), calibration);
            },
            "Empty image");
        requireThrows(
            [&] {
                lens_distortion::undistortImage(
                    sample, calibration, -0.1);
            },
            "Invalid alpha");
        requireThrows(
            [&] {
                lens_distortion::undistortImage(
                    sample, calibration, 1.0, "unknown");
            },
            "Invalid method");
        requireThrows(
            [&] {
                lens_distortion::undistortImage(
                    sample(cv::Rect(0, 0, 100, 100)),
                    calibration);
            },
            "Mismatched image dimensions");

        std::cout
            << "Lens-distortion tests passed: 41/41 images, RMS "
            << calibration.rms << ", ROI " << direct.roi << ".\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << "\n";
        return 1;
    }
}
