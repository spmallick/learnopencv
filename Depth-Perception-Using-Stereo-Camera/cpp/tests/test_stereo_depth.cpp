#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

#include "stereo_depth.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(
    double actual, double expected, double tolerance,
    const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

learnopencv::stereo::StereoBMConfig syntheticConfig() {
    learnopencv::stereo::StereoBMConfig config;
    config.numDisparities = 16;
    config.blockSize = 9;
    config.preFilterType = 1;
    config.preFilterSize = 9;
    config.preFilterCap = 31;
    config.textureThreshold = 0;
    config.uniquenessRatio = 0;
    config.speckleRange = 0;
    config.speckleWindowSize = 0;
    config.disp12MaxDiff = -1;
    config.minDisparity = 0;
    config.depthScale = 1000.0;
    config.depthOffset = 5.0;
    config.validate();
    return config;
}

std::pair<cv::Mat, cv::Mat> syntheticPair(int disparity = 8) {
    cv::RNG random(7);
    cv::Mat left(120, 192, CV_8UC1);
    random.fill(left, cv::RNG::UNIFORM, 0, 256);
    cv::Mat right = cv::Mat::zeros(left.size(), left.type());
    left.colRange(disparity, left.cols).copyTo(
        right.colRange(0, right.cols - disparity));
    random.fill(
        right.colRange(right.cols - disparity, right.cols),
        cv::RNG::UNIFORM, 0, 256);
    return {left, right};
}

}  // namespace

int main() {
    try {
        const auto source =
            std::filesystem::path(STEREO_SOURCE_DIR);
        const auto legacy = learnopencv::stereo::loadConfig(
            source / "data" / "depth_estimation_params_cpp.xml");
        require(legacy.numDisparities == 144,
                "Legacy numDisparities mismatch.");
        requireNear(
            legacy.depthScale, 41.32084274291992 * 144.0, 1e-8,
            "Legacy depth scale migration failed.");

        const auto maps =
            learnopencv::stereo::loadRectificationMaps(
                source / "data" / "stereo_rectify_maps.xml");
        require(
            maps.imageSize() == cv::Size(640, 480),
            "Rectification map size mismatch.");
        learnopencv::stereo::RectificationMaps invalidMaps;
        invalidMaps.leftX = cv::Mat::zeros(10, 10, CV_8UC1);
        invalidMaps.leftY = cv::Mat::zeros(10, 10, CV_8UC1);
        invalidMaps.rightX = cv::Mat::zeros(10, 10, CV_8UC1);
        invalidMaps.rightY = cv::Mat::zeros(10, 10, CV_8UC1);
        bool invalidMapsRejected = false;
        try {
            invalidMaps.validate();
        } catch (const std::invalid_argument&) {
            invalidMapsRejected = true;
        }
        require(
            invalidMapsRejected,
            "Invalid rectification map types were accepted.");

        auto invalid = syntheticConfig();
        invalid.numDisparities = 15;
        bool invalidRejected = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument&) {
            invalidRejected = true;
        }
        require(invalidRejected, "Invalid StereoBM config was accepted.");

        const auto config = syntheticConfig();
        const auto temporary =
            std::filesystem::current_path() /
            "stereo-depth-roundtrip.xml";
        learnopencv::stereo::saveConfig(config, temporary);
        const auto roundTrip =
            learnopencv::stereo::loadConfig(temporary);
        std::filesystem::remove(temporary);
        requireNear(
            roundTrip.depthScale, config.depthScale, 1e-12,
            "Config scale round trip failed.");
        requireNear(
            roundTrip.depthOffset, config.depthOffset, 1e-12,
            "Config offset round trip failed.");

        const auto [left, right] = syntheticPair();
        const cv::Mat disparity =
            learnopencv::stereo::computeDisparity(
                left, right, config);
        std::vector<float> finite;
        for (int row = 0; row < disparity.rows; ++row) {
            const auto* values = disparity.ptr<float>(row);
            for (int col = 0; col < disparity.cols; ++col) {
                if (std::isfinite(values[col])) {
                    finite.push_back(values[col]);
                }
            }
        }
        require(
            finite.size() >
                static_cast<std::size_t>(disparity.total() * 0.75),
            "Too few valid synthetic disparities.");
        std::sort(finite.begin(), finite.end());
        requireNear(
            finite[finite.size() / 2], 8.0, 0.25,
            "Synthetic disparity median mismatch.");

        const std::vector<double> disparities{
            8.0, 10.0, 16.0, 25.0, 40.0};
        std::vector<double> depths;
        for (const double value : disparities) {
            depths.push_back(1200.0 / (value - 2.0) + 17.0);
        }
        const auto fit = learnopencv::stereo::fitDepthModel(
            disparities, depths, 2.0);
        requireNear(fit.scale, 1200.0, 1e-8,
                    "Depth fit scale mismatch.");
        requireNear(fit.offset, 17.0, 1e-8,
                    "Depth fit offset mismatch.");
        require(fit.rmse < 1e-10, "Depth fit RMSE is too high.");

        auto depthConfig = config;
        depthConfig.minDisparity = 2;
        cv::Mat sample(
            1, 4, CV_32FC1,
            cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        sample.at<float>(0, 1) = -1.0F;
        sample.at<float>(0, 2) = 2.0F;
        sample.at<float>(0, 3) = 7.0F;
        const cv::Mat depth =
            learnopencv::stereo::disparityToDepth(
                sample, depthConfig);
        require(std::isnan(depth.at<float>(0, 0)),
                "NaN disparity must remain invalid.");
        require(std::isnan(depth.at<float>(0, 1)),
                "Negative disparity must be invalid.");
        require(std::isnan(depth.at<float>(0, 2)),
                "Minimum disparity must be invalid.");
        requireNear(depth.at<float>(0, 3), 205.0, 1e-5,
                    "Depth conversion mismatch.");

        cv::Mat obstacleDepth(
            100, 100, CV_32FC1,
            cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        obstacleDepth(
            cv::Rect(30, 20, 50, 50)).setTo(50.0F);
        obstacleDepth(
            cv::Rect(40, 30, 10, 10)).setTo(
                std::numeric_limits<float>::quiet_NaN());
        const auto [mask, obstacle] =
            learnopencv::stereo::findLargestObstacle(
                obstacleDepth, 10.0, 100.0, 0.1);
        require(obstacle.has_value(), "Expected obstacle was not found.");
        require(
            obstacle->boundingBox == cv::Rect(30, 20, 50, 50),
            "Obstacle bounding box mismatch.");
        requireNear(obstacle->meanDepth, 50.0, 1e-6,
                    "Obstacle mean depth mismatch.");
        require(cv::countNonZero(mask) == 2400,
                "Obstacle mask mismatch.");

        std::cout << "All stereo-depth C++ regression checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Stereo-depth regression failure: "
                  << error.what() << '\n';
        return 1;
    }
}
