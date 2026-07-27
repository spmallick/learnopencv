#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace lens_distortion {

struct CalibrationResult {
    double rms = 0.0;
    double reprojectionRMSE = 0.0;
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationVectors;
    cv::Size imageSize;
    std::vector<std::filesystem::path> successfulImages;
    std::vector<std::filesystem::path> failedImages;
    cv::Mat cornerPreview;
};

struct UndistortionResult {
    cv::Mat image;
    cv::Mat newCameraMatrix;
    cv::Rect roi;
};

std::vector<std::filesystem::path> discoverImages(
    const std::filesystem::path& directory);

CalibrationResult calibrateFromImages(
    const std::vector<std::filesystem::path>& imagePaths,
    cv::Size boardSize = cv::Size(6, 9),
    double squareSize = 1.0,
    bool requireAll = false);

UndistortionResult undistortImage(
    const cv::Mat& image,
    const CalibrationResult& calibration,
    double alpha = 1.0,
    const std::string& method = "direct",
    bool crop = false);

void saveCalibration(
    const CalibrationResult& calibration,
    const std::filesystem::path& destination);

}  // namespace lens_distortion
