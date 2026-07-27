#include "stereo_depth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

namespace learnopencv::stereo {
namespace {

double readRequired(const cv::FileStorage& storage, const char* name) {
    const cv::FileNode node = storage[name];
    if (node.empty()) {
        throw std::invalid_argument(
            std::string("Missing required configuration field: ") + name);
    }
    const double value = static_cast<double>(node);
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string("Configuration field is not finite: ") + name);
    }
    return value;
}

double readOptional(
    const cv::FileStorage& storage, const char* name, double fallback) {
    const cv::FileNode node = storage[name];
    return node.empty() ? fallback : static_cast<double>(node);
}

void validatePair(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        throw std::invalid_argument(
            "Left and right images must be non-empty.");
    }
    if (left.size() != right.size() || left.type() != right.type()) {
        throw std::invalid_argument(
            "Left and right images must have the same size and type.");
    }
    if (left.type() != CV_8UC1) {
        throw std::invalid_argument(
            "Stereo matching expects CV_8UC1 grayscale images.");
    }
}

}  // namespace

void StereoBMConfig::validate() const {
    if (numDisparities <= 0 || numDisparities % 16 != 0) {
        throw std::invalid_argument(
            "numDisparities must be a positive multiple of 16.");
    }
    if (blockSize < 5 || blockSize > 255 || blockSize % 2 == 0) {
        throw std::invalid_argument(
            "blockSize must be odd and in [5, 255].");
    }
    if (preFilterType != 0 && preFilterType != 1) {
        throw std::invalid_argument("preFilterType must be 0 or 1.");
    }
    if (preFilterSize < 5 || preFilterSize > 255 ||
        preFilterSize % 2 == 0) {
        throw std::invalid_argument(
            "preFilterSize must be odd and in [5, 255].");
    }
    if (preFilterCap < 1 || preFilterCap > 63) {
        throw std::invalid_argument("preFilterCap must be in [1, 63].");
    }
    if (textureThreshold < 0 || uniquenessRatio < 0 ||
        speckleRange < 0 || speckleWindowSize < 0) {
        throw std::invalid_argument(
            "StereoBM filtering values must be non-negative.");
    }
    if (disp12MaxDiff < -1) {
        throw std::invalid_argument("disp12MaxDiff must be -1 or greater.");
    }
    if (!std::isfinite(depthScale) || depthScale <= 0.0) {
        throw std::invalid_argument(
            "depthScale must be finite and positive.");
    }
    if (!std::isfinite(depthOffset)) {
        throw std::invalid_argument("depthOffset must be finite.");
    }
}

void RectificationMaps::validate() const {
    if (leftX.empty() || leftY.empty() || rightX.empty() ||
        rightY.empty()) {
        throw std::invalid_argument(
            "One or more rectification maps are empty.");
    }
    if (leftX.size() != leftY.size() ||
        rightX.size() != rightY.size() ||
        leftX.size() != rightX.size()) {
        throw std::invalid_argument(
            "Rectification map image sizes do not match.");
    }
    const auto validPair = [](const cv::Mat& mapX, const cv::Mat& mapY) {
        return
            (mapX.type() == CV_16SC2 && mapY.type() == CV_16UC1) ||
            (mapX.type() == CV_32FC1 && mapY.type() == CV_32FC1);
    };
    if (!validPair(leftX, leftY) || !validPair(rightX, rightY)) {
        throw std::invalid_argument(
            "Rectification maps must use OpenCV fixed-point or "
            "single-channel floating-point map pairs.");
    }
}

cv::Size RectificationMaps::imageSize() const {
    validate();
    return leftX.size();
}

StereoBMConfig loadConfig(const std::filesystem::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error(
            "Could not open stereo configuration: " + path.string());
    }

    StereoBMConfig config;
    config.numDisparities =
        static_cast<int>(readRequired(storage, "numDisparities"));
    config.blockSize = static_cast<int>(readRequired(storage, "blockSize"));
    config.preFilterType =
        static_cast<int>(readRequired(storage, "preFilterType"));
    config.preFilterSize =
        static_cast<int>(readRequired(storage, "preFilterSize"));
    config.preFilterCap =
        static_cast<int>(readRequired(storage, "preFilterCap"));
    config.textureThreshold =
        static_cast<int>(readRequired(storage, "textureThreshold"));
    config.uniquenessRatio =
        static_cast<int>(readRequired(storage, "uniquenessRatio"));
    config.speckleRange =
        static_cast<int>(readRequired(storage, "speckleRange"));
    config.speckleWindowSize =
        static_cast<int>(readRequired(storage, "speckleWindowSize"));
    config.disp12MaxDiff =
        static_cast<int>(readRequired(storage, "disp12MaxDiff"));
    config.minDisparity =
        static_cast<int>(readRequired(storage, "minDisparity"));

    if (storage["depthScale"].empty()) {
        const double legacyScale = readRequired(storage, "M");
        config.depthScale =
            legacyScale * static_cast<double>(config.numDisparities);
    } else {
        config.depthScale = readRequired(storage, "depthScale");
    }
    config.depthOffset = readOptional(
        storage, "depthOffset", readOptional(storage, "C", 0.0));
    config.validate();
    return config;
}

void saveConfig(
    const StereoBMConfig& config, const std::filesystem::path& path) {
    config.validate();
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    cv::FileStorage storage(path.string(), cv::FileStorage::WRITE);
    if (!storage.isOpened()) {
        throw std::runtime_error(
            "Could not open configuration for writing: " + path.string());
    }
    storage << "numDisparities" << config.numDisparities;
    storage << "blockSize" << config.blockSize;
    storage << "preFilterType" << config.preFilterType;
    storage << "preFilterSize" << config.preFilterSize;
    storage << "preFilterCap" << config.preFilterCap;
    storage << "textureThreshold" << config.textureThreshold;
    storage << "uniquenessRatio" << config.uniquenessRatio;
    storage << "speckleRange" << config.speckleRange;
    storage << "speckleWindowSize" << config.speckleWindowSize;
    storage << "disp12MaxDiff" << config.disp12MaxDiff;
    storage << "minDisparity" << config.minDisparity;
    storage << "depthScale" << config.depthScale;
    storage << "depthOffset" << config.depthOffset;
}

RectificationMaps loadRectificationMaps(
    const std::filesystem::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error(
            "Could not open rectification maps: " + path.string());
    }
    RectificationMaps maps;
    storage["Left_Stereo_Map_x"] >> maps.leftX;
    storage["Left_Stereo_Map_y"] >> maps.leftY;
    storage["Right_Stereo_Map_x"] >> maps.rightX;
    storage["Right_Stereo_Map_y"] >> maps.rightY;
    maps.validate();
    return maps;
}

cv::Ptr<cv::StereoBM> createMatcher(const StereoBMConfig& config) {
    config.validate();
    cv::Ptr<cv::StereoBM> matcher =
        cv::StereoBM::create(config.numDisparities, config.blockSize);
    matcher->setPreFilterType(config.preFilterType);
    matcher->setPreFilterSize(config.preFilterSize);
    matcher->setPreFilterCap(config.preFilterCap);
    matcher->setTextureThreshold(config.textureThreshold);
    matcher->setUniquenessRatio(config.uniquenessRatio);
    matcher->setSpeckleRange(config.speckleRange);
    matcher->setSpeckleWindowSize(config.speckleWindowSize);
    matcher->setDisp12MaxDiff(config.disp12MaxDiff);
    matcher->setMinDisparity(config.minDisparity);
    return matcher;
}

std::pair<cv::Mat, cv::Mat> rectifyPair(
    const cv::Mat& left,
    const cv::Mat& right,
    const RectificationMaps& maps) {
    validatePair(left, right);
    maps.validate();
    if (left.size() != maps.imageSize()) {
        throw std::invalid_argument(
            "Stereo image size does not match the rectification maps.");
    }
    cv::Mat leftRectified;
    cv::Mat rightRectified;
    cv::remap(
        left, leftRectified, maps.leftX, maps.leftY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
    cv::remap(
        right, rightRectified, maps.rightX, maps.rightY,
        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
    return {leftRectified, rightRectified};
}

cv::Mat computeDisparity(
    const cv::Mat& left,
    const cv::Mat& right,
    const StereoBMConfig& config,
    const cv::Ptr<cv::StereoBM>& suppliedMatcher) {
    validatePair(left, right);
    config.validate();
    const cv::Ptr<cv::StereoBM> matcher =
        suppliedMatcher.empty() ? createMatcher(config) : suppliedMatcher;
    cv::Mat fixedPoint;
    matcher->compute(left, right, fixedPoint);
    cv::Mat disparity;
    fixedPoint.convertTo(disparity, CV_32F, 1.0 / 16.0);
    disparity.setTo(
        std::numeric_limits<float>::quiet_NaN(),
        disparity <= static_cast<float>(config.minDisparity));
    return disparity;
}

cv::Mat disparityToDepth(
    const cv::Mat& disparityPixels, const StereoBMConfig& config) {
    config.validate();
    if (disparityPixels.empty() || disparityPixels.type() != CV_32FC1) {
        throw std::invalid_argument(
            "Disparity must be a non-empty CV_32FC1 matrix.");
    }
    cv::Mat depth(
        disparityPixels.size(), CV_32FC1,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (int row = 0; row < disparityPixels.rows; ++row) {
        const auto* disparity = disparityPixels.ptr<float>(row);
        auto* output = depth.ptr<float>(row);
        for (int col = 0; col < disparityPixels.cols; ++col) {
            const double denominator =
                static_cast<double>(disparity[col]) -
                static_cast<double>(config.minDisparity);
            if (std::isfinite(denominator) && denominator > 0.0) {
                const double value =
                    config.depthScale / denominator + config.depthOffset;
                if (std::isfinite(value)) {
                    output[col] = static_cast<float>(value);
                }
            }
        }
    }
    return depth;
}

DepthFit fitDepthModel(
    const std::vector<double>& disparityPixels,
    const std::vector<double>& measuredDepths,
    double minDisparity) {
    if (disparityPixels.size() != measuredDepths.size()) {
        throw std::invalid_argument(
            "Disparity and depth samples must have matching sizes.");
    }

    std::vector<double> inverseDisparities;
    std::vector<double> validDepths;
    for (std::size_t index = 0; index < disparityPixels.size(); ++index) {
        const double denominator = disparityPixels[index] - minDisparity;
        if (std::isfinite(denominator) && denominator > 0.0 &&
            std::isfinite(measuredDepths[index])) {
            inverseDisparities.push_back(1.0 / denominator);
            validDepths.push_back(measuredDepths[index]);
        }
    }
    if (inverseDisparities.size() < 2) {
        throw std::invalid_argument(
            "At least two finite positive-disparity samples are required.");
    }

    const double count =
        static_cast<double>(inverseDisparities.size());
    const double xMean =
        std::accumulate(
            inverseDisparities.begin(), inverseDisparities.end(), 0.0) /
        count;
    const double yMean =
        std::accumulate(validDepths.begin(), validDepths.end(), 0.0) /
        count;

    double covariance = 0.0;
    double variance = 0.0;
    for (std::size_t index = 0; index < inverseDisparities.size();
         ++index) {
        const double xDelta = inverseDisparities[index] - xMean;
        covariance += xDelta * (validDepths[index] - yMean);
        variance += xDelta * xDelta;
    }
    if (variance <= std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument(
            "Depth samples do not define both scale and offset.");
    }

    DepthFit fit;
    fit.scale = covariance / variance;
    fit.offset = yMean - fit.scale * xMean;
    double squaredError = 0.0;
    for (std::size_t index = 0; index < inverseDisparities.size();
         ++index) {
        const double prediction =
            fit.scale * inverseDisparities[index] + fit.offset;
        const double error = prediction - validDepths[index];
        squaredError += error * error;
    }
    fit.rmse = std::sqrt(squaredError / count);
    return fit;
}

std::pair<cv::Mat, std::optional<Obstacle>> findLargestObstacle(
    const cv::Mat& depth,
    double minDepth,
    double maxDepth,
    double minimumAreaFraction) {
    if (depth.empty() || depth.type() != CV_32FC1) {
        throw std::invalid_argument(
            "Depth must be a non-empty CV_32FC1 matrix.");
    }
    if (!std::isfinite(minDepth) || !std::isfinite(maxDepth) ||
        minDepth >= maxDepth) {
        throw std::invalid_argument(
            "Depth limits must be finite and minDepth < maxDepth.");
    }
    if (minimumAreaFraction < 0.0 || minimumAreaFraction > 1.0) {
        throw std::invalid_argument(
            "minimumAreaFraction must be in [0, 1].");
    }

    cv::Mat mask;
    cv::inRange(depth, minDepth, maxDepth, mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        mask.clone(), contours, cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {mask, std::nullopt};
    }
    const auto largest = std::max_element(
        contours.begin(), contours.end(),
        [](const auto& first, const auto& second) {
            return cv::contourArea(first) < cv::contourArea(second);
        });
    const double area = std::abs(cv::contourArea(*largest));
    const double imageArea =
        static_cast<double>(depth.rows) * depth.cols;
    const double areaFraction = area / imageArea;
    if (areaFraction < minimumAreaFraction) {
        return {mask, std::nullopt};
    }

    cv::Mat contourMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(
        contourMask, contours,
        static_cast<int>(std::distance(contours.begin(), largest)),
        cv::Scalar(255), cv::FILLED);
    cv::Mat validContourMask;
    cv::bitwise_and(contourMask, mask, validContourMask);
    Obstacle obstacle;
    obstacle.boundingBox = cv::boundingRect(*largest);
    obstacle.meanDepth = cv::mean(depth, validContourMask)[0];
    obstacle.areaFraction = areaFraction;
    return {mask, obstacle};
}

cv::Mat disparityVisualization(const cv::Mat& disparity) {
    if (disparity.empty() || disparity.type() != CV_32FC1) {
        throw std::invalid_argument(
            "Disparity must be a non-empty CV_32FC1 matrix.");
    }
    cv::Mat validMask = disparity == disparity;
    cv::Mat output = cv::Mat::zeros(disparity.size(), CV_8UC1);
    if (cv::countNonZero(validMask) == 0) {
        return output;
    }
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(disparity, &minimum, &maximum, nullptr, nullptr, validMask);
    if (maximum <= minimum) {
        return output;
    }
    for (int row = 0; row < disparity.rows; ++row) {
        const auto* input = disparity.ptr<float>(row);
        auto* visual = output.ptr<std::uint8_t>(row);
        for (int col = 0; col < disparity.cols; ++col) {
            if (std::isfinite(input[col])) {
                const double scaled =
                    (static_cast<double>(input[col]) - minimum) * 255.0 /
                    (maximum - minimum);
                visual[col] = cv::saturate_cast<std::uint8_t>(scaled);
            }
        }
    }
    return output;
}

}  // namespace learnopencv::stereo
