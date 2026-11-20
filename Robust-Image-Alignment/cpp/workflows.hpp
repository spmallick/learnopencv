#pragma once

#include "alignment.hpp"

#include <filesystem>
#include <optional>

namespace robust_alignment {

// All paths can be supplied absolutely so experiments also work outside the checkout.
struct WorkflowOptions {
    std::filesystem::path project_dir;
    std::filesystem::path assets_dir;
    std::filesystem::path output_dir;
    std::filesystem::path model_dir;
    AlignmentConfig config;
    int max_keypoints = 2048;
    int resize = 1024;
    int repeats = 3;
    int study_seeds = 3;
};

// These errors use independent dataset geometry, not the estimator's chosen inliers.
struct GeometricError {
    bool has_model = false;
    std::optional<double> grid_median_error_px;
    std::optional<double> grid_p95_error_px;
    std::optional<double> corner_mean_error_px;
    int grid_points = 0;
    void write(cv::FileStorage& storage) const;
};

GeometricError geometric_error(const cv::Mat& moving_to_reference,
                               const cv::Mat& truth_reference_to_moving,
                               cv::Size reference_size, cv::Size moving_size);

// Demo validation covers known positives, known negatives, catalogue ambiguity,
// and the small-support exercise. Parallax and ECC remain diagnostic examples.
bool run_demo_all(const WorkflowOptions& options);

// The benchmark preserves failed estimates and counts image pairs only once.
bool run_benchmark(const WorkflowOptions& options);

// The native random stream is reproducible but differs from NumPy's PCG64 stream.
bool run_synthetic_study(const WorkflowOptions& options);

}  // namespace robust_alignment
