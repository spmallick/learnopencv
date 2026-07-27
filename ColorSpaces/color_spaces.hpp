#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

namespace color_spaces {

enum class ColorSpace {
    Bgr,
    Hsv,
    YCrCb,
    Lab,
};

inline std::string name(const ColorSpace colorSpace) {
    switch (colorSpace) {
        case ColorSpace::Bgr:
            return "BGR";
        case ColorSpace::Hsv:
            return "HSV";
        case ColorSpace::YCrCb:
            return "YCrCb";
        case ColorSpace::Lab:
            return "Lab";
    }
    throw std::logic_error("unreachable color-space value");
}

inline ColorSpace parseColorSpace(const std::string& value) {
    if (value == "BGR") {
        return ColorSpace::Bgr;
    }
    if (value == "HSV") {
        return ColorSpace::Hsv;
    }
    if (value == "YCrCb") {
        return ColorSpace::YCrCb;
    }
    if (value == "Lab") {
        return ColorSpace::Lab;
    }
    throw std::invalid_argument("unsupported color space: " + value);
}

inline cv::Mat readBgr(const std::filesystem::path& path) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("could not read image: " + path.string());
    }
    return image;
}

inline cv::Mat convertBgr(const cv::Mat& image, const ColorSpace colorSpace) {
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("image must be a nonempty 8-bit, three-channel BGR image");
    }
    if (colorSpace == ColorSpace::Bgr) {
        return image.clone();
    }

    int conversionCode = cv::COLOR_BGR2HSV;
    if (colorSpace == ColorSpace::YCrCb) {
        conversionCode = cv::COLOR_BGR2YCrCb;
    } else if (colorSpace == ColorSpace::Lab) {
        conversionCode = cv::COLOR_BGR2Lab;
    }

    cv::Mat converted;
    cv::cvtColor(image, converted, conversionCode);
    return converted;
}

struct PixelValues {
    cv::Vec3b bgr;
    cv::Vec3b hsv;
    cv::Vec3b yCrCb;
    cv::Vec3b lab;
};

inline PixelValues convertPixel(const cv::Vec3b& bgrPixel) {
    cv::Mat pixel(1, 1, CV_8UC3);
    pixel.at<cv::Vec3b>(0, 0) = bgrPixel;
    const cv::Vec3b bgr = bgrPixel;
    const cv::Vec3b hsv = convertBgr(pixel, ColorSpace::Hsv).at<cv::Vec3b>(0, 0);
    const cv::Vec3b yCrCb = convertBgr(pixel, ColorSpace::YCrCb).at<cv::Vec3b>(0, 0);
    const cv::Vec3b lab = convertBgr(pixel, ColorSpace::Lab).at<cv::Vec3b>(0, 0);
    return PixelValues{bgr, hsv, yCrCb, lab};
}

inline void validateThreshold(
    const std::array<int, 3>& values,
    const std::string& label) {
    for (const int value : values) {
        if (value < 0 || value > 255) {
            throw std::invalid_argument(label + " threshold values must be in [0, 255]");
        }
    }
}

inline cv::Scalar asScalar(const std::array<int, 3>& values) {
    return cv::Scalar(values[0], values[1], values[2]);
}

inline cv::Mat thresholdMask(
    const cv::Mat& imageBgr,
    const ColorSpace colorSpace,
    const std::array<int, 3>& lower,
    const std::array<int, 3>& upper) {
    validateThreshold(lower, "lower");
    validateThreshold(upper, "upper");
    const cv::Mat converted = convertBgr(imageBgr, colorSpace);

    if (colorSpace == ColorSpace::Hsv) {
        if (lower[0] > 179 || upper[0] > 179) {
            throw std::invalid_argument("8-bit HSV hue thresholds must be in [0, 179]");
        }
        if (lower[1] > upper[1] || lower[2] > upper[2]) {
            throw std::invalid_argument(
                "HSV saturation/value lower bounds must not exceed upper bounds");
        }
        if (lower[0] > upper[0]) {
            std::array<int, 3> firstUpper = upper;
            firstUpper[0] = 179;
            std::array<int, 3> secondLower = lower;
            secondLower[0] = 0;

            cv::Mat firstMask;
            cv::Mat secondMask;
            cv::inRange(converted, asScalar(lower), asScalar(firstUpper), firstMask);
            cv::inRange(converted, asScalar(secondLower), asScalar(upper), secondMask);

            cv::Mat combinedMask;
            cv::bitwise_or(firstMask, secondMask, combinedMask);
            return combinedMask;
        }
    } else {
        for (std::size_t index = 0; index < lower.size(); ++index) {
            if (lower[index] > upper[index]) {
                throw std::invalid_argument(
                    "lower threshold values must not exceed upper values");
            }
        }
    }

    cv::Mat mask;
    cv::inRange(converted, asScalar(lower), asScalar(upper), mask);
    return mask;
}

inline cv::Mat applyMask(const cv::Mat& imageBgr, const cv::Mat& mask) {
    if (mask.empty() || mask.type() != CV_8UC1 || mask.size() != imageBgr.size()) {
        throw std::invalid_argument("mask must be uint8 and match the image size");
    }
    cv::Mat result;
    cv::bitwise_and(imageBgr, imageBgr, result, mask);
    return result;
}

inline void writeImage(const std::filesystem::path& path, const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("cannot write an empty image");
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("could not write image: " + path.string());
    }
}

inline std::string vectorText(const cv::Vec3b& value) {
    std::ostringstream stream;
    stream << '[' << static_cast<int>(value[0]) << ", "
           << static_cast<int>(value[1]) << ", "
           << static_cast<int>(value[2]) << ']';
    return stream.str();
}

}  // namespace color_spaces
