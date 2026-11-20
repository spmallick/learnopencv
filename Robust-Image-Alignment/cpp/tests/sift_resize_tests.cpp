#include "feature_matcher.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    // Use a nonintegral scale on the shorter side to distinguish pixel-center geometry
    // from the ratio between rounded output dimensions and original dimensions.
    cv::Mat image(479, 733, CV_8UC3);
    cv::RNG random(31);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.6);
    robust_alignment::FeatureMatcher::Options options;
    options.method = "sift";
    options.resize = 181;
    const double scale = static_cast<double>(options.resize) / image.cols;
    const auto original = robust_alignment::FeatureMatcher(options).extract(image);
    cv::Mat work;
    cv::resize(image, work, cv::Size(), scale, scale);
    options.resize = 0;
    const auto direct = robust_alignment::FeatureMatcher(options).extract(work);
    if (direct.keypoints.size() < 8 || direct.keypoints.size() != original.keypoints.size() ||
        cv::norm(direct.descriptors, original.descriptors, cv::NORM_INF) != 0) {
        throw std::runtime_error("Both paths must detect the same SIFT features in the working image");
    }
    // Resizing a linear coordinate field gives a ground-truth map back to source
    // pixel centers without reimplementing the index formula used by FeatureMatcher.
    for (int axis = 0; axis < 2; ++axis) {
        cv::Mat coordinate(image.size(), CV_32F);
        for (int y = 0; y < image.rows; ++y) {
            float* row = coordinate.ptr<float>(y);
            for (int x = 0; x < image.cols; ++x) {
                row[x] = static_cast<float>(axis == 0 ? x : y);
            }
        }
        cv::Mat ramp;
        cv::resize(coordinate, ramp, cv::Size(), scale, scale);
        for (std::size_t index = 0; index < direct.keypoints.size(); ++index) {
            cv::Mat value;
            cv::getRectSubPix(ramp, cv::Size(1, 1), direct.keypoints[index], value);
            const auto point = original.keypoints[index];
            const float measured = axis == 0 ? point.x : point.y;
            if (std::abs(measured - value.at<float>(0, 0)) > 2e-4F) {
                throw std::runtime_error("The reported SIFT position differs from its source pixel center");
            }
        }
    }
    std::cout << "VALIDATION PASSED: SIFT pixel centers after a fractional resize\n";
    return 0;
}
