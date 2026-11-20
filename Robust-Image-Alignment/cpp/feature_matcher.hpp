#pragma once

#include "alignment.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace robust_alignment {

// Report the loaded native library version rather than a build-time package label.
std::string native_inference_runtime_version();

// The CPU reference contract uses these fixed ONNX Runtime thread counts.
inline constexpr int native_inference_intra_op_threads = 4;
inline constexpr int native_inference_inter_op_threads = 1;

// A cached feature set contains original-image pixel-center coordinates.
// Descriptors are contiguous N-by-128 floats for both ALIKED and SIFT.
struct FeatureSet {
    std::vector<cv::Point2f> keypoints;
    cv::Mat descriptors;
    cv::Size image_size;
    double extraction_ms = 0.0;
    std::string method;
};

// OpenCV supplies images, SIFT and nearest neighbors; native ONNX Runtime
// evaluates the same ALIKED-n16 and LightGlue weights as the Python companion.
// The implementation never starts a Python process or loads a Python runtime.
class FeatureMatcher {
public:
    struct Options {
        std::string method = "lightglue";
        std::filesystem::path model_dir;
        int max_keypoints = 2048;
        int resize = 1024;
        float ratio = 0.8F;
    };

    explicit FeatureMatcher(Options options);
    ~FeatureMatcher();
    FeatureMatcher(FeatureMatcher&&) noexcept;
    FeatureMatcher& operator=(FeatureMatcher&&) noexcept;
    FeatureMatcher(const FeatureMatcher&) = delete;
    FeatureMatcher& operator=(const FeatureMatcher&) = delete;

    FeatureSet extract(const cv::Mat& image_bgr);
    MatchResult match(const FeatureSet& reference, const FeatureSet& moving);
    MatchResult match_images(const cv::Mat& reference_bgr, const cv::Mat& moving_bgr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace robust_alignment
