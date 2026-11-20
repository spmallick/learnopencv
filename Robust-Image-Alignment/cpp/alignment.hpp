#pragma once

// This geometry layer is independent of the feature-inference backend.
#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace robust_alignment {

// Coordinates always refer to the ORIGINAL images, in OpenCV's (x,y) convention.
struct MatchResult {
    std::vector<cv::Point2f> reference_points;
    std::vector<cv::Point2f> moving_points;
    std::vector<float> scores;
    std::string method = "provided";
    int reference_keypoints = 0;
    int moving_keypoints = 0;
    double extraction_ms = 0.0;
    double matching_ms = 0.0;

    // Reject inconsistent arrays before a robust estimator can read them.
    void validate() const;
};

// These gates match the Python companion; they are heuristics, not identity probabilities.
struct AlignmentConfig {
    double reprojection_threshold = 3.0;
    double confidence = 0.999;
    int max_iterations = 10000;
    int min_inliers = 12;
    double min_inlier_ratio = 0.25;
    double min_reference_coverage = 0.04;
    double max_median_reprojection_error = 3.0;
    double min_projected_area = 16.0;
    double min_visible_fraction = 0.10;

    // Validate user-supplied thresholds independently of whether matching succeeded.
    void validate() const;
    void write(cv::FileStorage& storage) const;
};

struct AlignmentResult {
    cv::Mat H;  // Empty means no estimate; otherwise CV_64F, moving -> reference.
    std::vector<std::uint8_t> inliers;
    bool accepted = false;
    std::vector<std::string> reasons;
    std::vector<cv::Point2f> projected_reference_quad;
    std::optional<double> median_reprojection_error;
    double reference_coverage = 0.0;
    double moving_coverage = 0.0;
    double visible_fraction = 0.0;
    double estimation_ms = 0.0;
    std::string estimator = "magsac";
    std::size_t num_matches = 0;

    std::size_t num_inliers() const;
    double inlier_ratio() const;
    // FileStorage writes fields within a mapping opened and closed by the caller.
    void write(cv::FileStorage& storage) const;
    // This form preserves Python's true/false and null representations exactly.
    std::string to_json() const;
};

cv::Mat load_image(const std::filesystem::path& path);
std::vector<cv::Point2f> image_corners(cv::Size image_size);
std::vector<cv::Point2f> project_points(const std::vector<cv::Point2f>& points, const cv::Mat& H);
double hull_coverage(const std::vector<cv::Point2f>& points, cv::Size image_size);

AlignmentResult estimate_alignment(
    const MatchResult& matches, cv::Size reference_size, cv::Size moving_size,
    const AlignmentConfig& config = AlignmentConfig{}, const std::string& estimator = "magsac");

// The nearest-pixel mask is convenient for display. Full support excludes output
// pixels whose bilinear intensity receives any nonzero contribution from the border.
std::pair<cv::Mat, cv::Mat> warp_alignment(
    const cv::Mat& moving_bgr, const AlignmentResult& result, cv::Size reference_size,
    bool require_accepted = true, bool full_support = false);

std::pair<cv::Mat, double> refine_ecc(
    const cv::Mat& reference_bgr, const cv::Mat& moving_bgr, const cv::Mat& H_moving_to_reference,
    int iterations = 100, double epsilon = 1e-6);

cv::Mat draw_matches(
    const cv::Mat& reference_bgr, const cv::Mat& moving_bgr, const MatchResult& matches,
    const AlignmentResult* result = nullptr, int max_draw = 100);

}  // namespace robust_alignment
