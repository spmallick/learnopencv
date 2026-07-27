#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace hough {

inline cv::Mat readBgr(const std::filesystem::path& path) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("could not read image: " + path.string());
    }
    return image;
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

inline cv::Vec4i normalizeLine(cv::Vec4i line) {
    const std::pair<int, int> first(line[0], line[1]);
    const std::pair<int, int> second(line[2], line[3]);
    if (second < first) {
        std::swap(line[0], line[2]);
        std::swap(line[1], line[3]);
    }
    return line;
}

struct LineDetection {
    cv::Mat edges;
    std::vector<cv::Vec4i> lines;
};

inline LineDetection detectLines(
    const cv::Mat& imageBgr,
    const int cannyLow = 50,
    const int cannyHigh = 150,
    const int houghThreshold = 50,
    const int minimumLineLength = 40,
    const int maximumLineGap = 25) {
    if (imageBgr.empty() || imageBgr.type() != CV_8UC3) {
        throw std::invalid_argument("line input must be a nonempty BGR image");
    }
    if (cannyLow < 0 || cannyLow >= cannyHigh) {
        throw std::invalid_argument("Canny thresholds must satisfy 0 <= low < high");
    }
    if (houghThreshold <= 0 || minimumLineLength < 0 || maximumLineGap < 0) {
        throw std::invalid_argument(
            "Hough threshold/length/gap parameters must be nonnegative");
    }

    cv::Mat gray;
    cv::cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);
    cv::Mat edges;
    cv::Canny(blurred, edges, cannyLow, cannyHigh);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(
        edges,
        lines,
        1.0,
        CV_PI / 180.0,
        houghThreshold,
        static_cast<double>(minimumLineLength),
        static_cast<double>(maximumLineGap));
    for (cv::Vec4i& line : lines) {
        line = normalizeLine(line);
    }
    std::sort(
        lines.begin(),
        lines.end(),
        [](const cv::Vec4i& left, const cv::Vec4i& right) {
            return std::tie(left[0], left[1], left[2], left[3]) <
                   std::tie(right[0], right[1], right[2], right[3]);
        });
    return LineDetection{edges, lines};
}

inline cv::Mat drawLines(
    const cv::Mat& imageBgr,
    const std::vector<cv::Vec4i>& lines) {
    cv::Mat output = imageBgr.clone();
    for (const cv::Vec4i& line : lines) {
        cv::line(
            output,
            cv::Point(line[0], line[1]),
            cv::Point(line[2], line[3]),
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA);
    }
    return output;
}

struct CircleDetection {
    cv::Mat blurred;
    std::vector<cv::Vec3f> circles;
};

inline CircleDetection detectCircles(
    const cv::Mat& imageBgr,
    const double dp = 1.2,
    const double minimumDistance = 20.0,
    const double param1 = 120.0,
    const double param2 = 30.0,
    const int minimumRadius = 20,
    const int maximumRadius = 60) {
    if (imageBgr.empty() || imageBgr.type() != CV_8UC3) {
        throw std::invalid_argument("circle input must be a nonempty BGR image");
    }
    if (dp < 1.0 || minimumDistance <= 0.0) {
        throw std::invalid_argument("dp must be >= 1 and minimum distance positive");
    }
    if (param1 <= 0.0 || param2 <= 0.0) {
        throw std::invalid_argument("param1 and param2 must be positive");
    }
    if (minimumRadius < 0 || maximumRadius < minimumRadius) {
        throw std::invalid_argument("radius bounds must satisfy 0 <= min <= max");
    }

    cv::Mat gray;
    cv::cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::medianBlur(gray, blurred, 5);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(
        blurred,
        circles,
        cv::HOUGH_GRADIENT,
        dp,
        minimumDistance,
        param1,
        param2,
        minimumRadius,
        maximumRadius);
    std::sort(
        circles.begin(),
        circles.end(),
        [](const cv::Vec3f& left, const cv::Vec3f& right) {
            return std::tie(left[0], left[1], left[2]) <
                   std::tie(right[0], right[1], right[2]);
        });
    return CircleDetection{blurred, circles};
}

inline cv::Mat drawCircles(
    const cv::Mat& imageBgr,
    const std::vector<cv::Vec3f>& circles) {
    cv::Mat output = imageBgr.clone();
    for (const cv::Vec3f& circleValue : circles) {
        const cv::Point center(
            cvRound(circleValue[0]),
            cvRound(circleValue[1]));
        const int radius = cvRound(circleValue[2]);
        cv::circle(
            output,
            center,
            radius,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA);
        cv::circle(
            output,
            center,
            3,
            cv::Scalar(0, 0, 255),
            -1,
            cv::LINE_AA);
    }
    return output;
}

}  // namespace hough
