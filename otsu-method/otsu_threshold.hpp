#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace learnopencv::otsu {

inline void validateGrayscale(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(
            "Otsu thresholding expects a non-empty CV_8UC1 image.");
    }
}

inline int threshold(const cv::Mat& image) {
    validateGrayscale(image);

    std::array<std::uint64_t, 256> histogram{};
    for (int row = 0; row < image.rows; ++row) {
        const auto* pixels = image.ptr<std::uint8_t>(row);
        for (int col = 0; col < image.cols; ++col) {
            ++histogram[pixels[col]];
        }
    }

    const auto totalPixels =
        static_cast<std::uint64_t>(image.rows) *
        static_cast<std::uint64_t>(image.cols);

    long double totalIntensity = 0.0L;
    for (std::size_t intensity = 0; intensity < histogram.size();
         ++intensity) {
        totalIntensity += static_cast<long double>(intensity) *
                          static_cast<long double>(histogram[intensity]);
    }

    std::uint64_t backgroundWeight = 0;
    long double backgroundIntensity = 0.0L;
    long double bestVariance = -1.0L;
    int bestThreshold = 0;

    for (int candidate = 0; candidate < 256; ++candidate) {
        const auto count = histogram[static_cast<std::size_t>(candidate)];
        backgroundWeight += count;
        backgroundIntensity += static_cast<long double>(candidate) *
                               static_cast<long double>(count);

        const auto foregroundWeight = totalPixels - backgroundWeight;
        if (backgroundWeight == 0) {
            continue;
        }
        if (foregroundWeight == 0) {
            break;
        }

        const long double backgroundMean =
            backgroundIntensity / static_cast<long double>(backgroundWeight);
        const long double foregroundMean =
            (totalIntensity - backgroundIntensity) /
            static_cast<long double>(foregroundWeight);
        const long double difference = backgroundMean - foregroundMean;
        const long double variance =
            static_cast<long double>(backgroundWeight) *
            static_cast<long double>(foregroundWeight) * difference *
            difference;

        // OpenCV resolves a plateau by keeping the first (lowest) threshold.
        if (variance > bestVariance) {
            bestVariance = variance;
            bestThreshold = candidate;
        }
    }

    return bestThreshold;
}

inline cv::Mat apply(const cv::Mat& image, int selectedThreshold) {
    validateGrayscale(image);
    cv::Mat binary;
    cv::threshold(
        image, binary, static_cast<double>(selectedThreshold), 255.0,
        cv::THRESH_BINARY);
    return binary;
}

}  // namespace learnopencv::otsu
