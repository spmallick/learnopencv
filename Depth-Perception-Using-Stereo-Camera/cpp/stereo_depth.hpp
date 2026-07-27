#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/stereo.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif

namespace learnopencv::stereo {

struct StereoBMConfig {
    int numDisparities = 144;
    int blockSize = 39;
    int preFilterType = 1;
    int preFilterSize = 9;
    int preFilterCap = 48;
    int textureThreshold = 9;
    int uniquenessRatio = 16;
    int speckleRange = 6;
    int speckleWindowSize = 24;
    int disp12MaxDiff = 0;
    int minDisparity = 0;
    double depthScale = 5950.201354980469;
    double depthOffset = 0.0;

    void validate() const;
};

struct RectificationMaps {
    cv::Mat leftX;
    cv::Mat leftY;
    cv::Mat rightX;
    cv::Mat rightY;

    void validate() const;
    cv::Size imageSize() const;
};

struct DepthFit {
    double scale = 0.0;
    double offset = 0.0;
    double rmse = 0.0;
};

struct Obstacle {
    cv::Rect boundingBox;
    double meanDepth = 0.0;
    double areaFraction = 0.0;
};

StereoBMConfig loadConfig(const std::filesystem::path& path);
void saveConfig(
    const StereoBMConfig& config, const std::filesystem::path& path);
RectificationMaps loadRectificationMaps(const std::filesystem::path& path);
cv::Ptr<cv::StereoBM> createMatcher(const StereoBMConfig& config);

std::pair<cv::Mat, cv::Mat> rectifyPair(
    const cv::Mat& left,
    const cv::Mat& right,
    const RectificationMaps& maps);
cv::Mat computeDisparity(
    const cv::Mat& left,
    const cv::Mat& right,
    const StereoBMConfig& config,
    const cv::Ptr<cv::StereoBM>& matcher = {});
cv::Mat disparityToDepth(
    const cv::Mat& disparityPixels, const StereoBMConfig& config);
DepthFit fitDepthModel(
    const std::vector<double>& disparityPixels,
    const std::vector<double>& measuredDepths,
    double minDisparity);
std::pair<cv::Mat, std::optional<Obstacle>> findLargestObstacle(
    const cv::Mat& depth,
    double minDepth,
    double maxDepth,
    double minimumAreaFraction = 0.01);
cv::Mat disparityVisualization(const cv::Mat& disparity);

}  // namespace learnopencv::stereo
