#include "alignment.hpp"

#include <opencv2/core.hpp>
#include <opencv2/geometry/3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Throwing checks remain active in Release builds, unlike the C assert macro.
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Callable>
void require_throws(Callable operation, const std::string& message,
                    const std::string& expected_fragment = "") {
    try {
        operation();
    } catch (const std::exception& error) {
        // Checking the message distinguishes the intended guard from an unrelated error.
        require(expected_fragment.empty() || std::string(error.what()).find(expected_fragment) != std::string::npos,
                message + ": unexpected exception: " + error.what());
        return;
    }
    throw std::runtime_error(message + ": no exception was raised");
}

bool has_reason(const robust_alignment::AlignmentResult& result, const std::string& fragment) {
    for (const std::string& reason : result.reasons) {
        if (reason.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

cv::Mat textured_image(cv::Size size = cv::Size(400, 320)) {
    // Original deterministic test pixels avoid a network or third-party image dependency.
    cv::Mat image(size, CV_8UC3);
    cv::RNG random(4);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.6);
    cv::putText(image, "REFERENCE 42", cv::Point(25, 140), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(10, 10, 10), 3);
    cv::rectangle(image, cv::Rect(40, 180, 130, 105), cv::Scalar(200, 40, 20), 3);
    return image;
}

robust_alignment::MatchResult known_matches(const cv::Mat& reference_to_moving) {
    robust_alignment::MatchResult matches;
    // A regular grid supplies independent support over most of the reference surface.
    for (int y = 30; y < 290; y += 26) {
        for (int x = 30; x < 370; x += 34) {
            matches.reference_points.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    matches.moving_points = robust_alignment::project_points(matches.reference_points, reference_to_moving);
    return matches;
}

double corner_error(const cv::Mat& reference_to_moving, const cv::Mat& moving_to_reference,
                    cv::Size size) {
    const auto corners = robust_alignment::image_corners(size);
    // These corners are independent of the correspondences supplied to the estimator.
    const auto moving = robust_alignment::project_points(corners, reference_to_moving);
    const auto recovered = robust_alignment::project_points(moving, moving_to_reference);
    double maximum = 0.0;
    for (std::size_t index = 0; index < corners.size(); ++index) {
        maximum = std::max(maximum, cv::norm(corners[index] - recovered[index]));
    }
    return maximum;
}

void recovers_known_transform_despite_outliers() {
    const cv::Size size(400, 320);
    const cv::Mat forward(cv::Matx33d(0.94, -0.08, 30, 0.04, 0.96, 8, 0.00012, -0.00008, 1));
    auto matches = known_matches(forward);
    // Thirty obvious wrong matches test estimation, not a trivial all-inlier fit.
    for (std::size_t index = matches.moving_points.size() - 30; index < matches.moving_points.size(); ++index) {
        matches.moving_points[index] = cv::Point2f(800.0F + static_cast<float>(index) * 13.0F,
                                                  600.0F + static_cast<float>((index * index) % 200));
    }
    for (const std::string estimator : {"ransac", "magsac"}) {
        const auto result = robust_alignment::estimate_alignment(matches, size, size, {}, estimator);
        require(result.accepted, estimator + " rejected a well-supported known transformation");
        require(result.num_inliers() >= 70, estimator + " lost the known correct correspondences");
        require(corner_error(forward, result.H, size) < 0.05,
                estimator + " does not map independent points back into reference coordinates");
        const auto expected_quad = robust_alignment::project_points(robust_alignment::image_corners(size), forward);
        for (std::size_t index = 0; index < expected_quad.size(); ++index) {
            require(cv::norm(expected_quad[index] - result.projected_reference_quad[index]) < 0.05,
                    estimator + " projects the reference boundary in the wrong direction");
        }
    }
}

void accepts_a_small_target_inside_a_large_scene() {
    robust_alignment::MatchResult matches;
    for (int y = 15; y < 150; y += 25) {
        for (int x = 15; x < 190; x += 30) {
            matches.reference_points.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    const cv::Mat forward(cv::Matx33d(0.5, 0, 500, 0, 0.5, 300, 0, 0, 1));
    matches.moving_points = robust_alignment::project_points(matches.reference_points, forward);
    const auto result = robust_alignment::estimate_alignment(matches, cv::Size(200, 160), cv::Size(1200, 1000));
    require(result.accepted, "Small objects must not fail merely because scene coverage is small");
    require(result.moving_coverage < 0.01, "Fixture did not put a small target in a large scene");
    require(result.reference_coverage > 0.5, "The target itself should have broad inlier support");
}

void rejects_missing_and_collinear_correspondences() {
    robust_alignment::MatchResult empty;
    const auto missing = robust_alignment::estimate_alignment(empty, cv::Size(100, 100), cv::Size(100, 100));
    require(!missing.accepted && missing.H.empty(), "Empty matches must produce an explicit rejection");
    require(has_reason(missing, "Fewer than four"), "Missing-match rejection did not explain its cause");
    robust_alignment::MatchResult line;
    for (int x = 0; x < 20; ++x) {
        line.reference_points.emplace_back(static_cast<float>(x), 10.0F);
    }
    line.moving_points = line.reference_points;
    const auto collinear = robust_alignment::estimate_alignment(line, cv::Size(100, 100), cv::Size(100, 100));
    require(!collinear.accepted && collinear.H.empty(), "Collinear matches cannot constrain a homography");
    require(has_reason(collinear, "collinear"), "Collinearity was not identified before estimation");
}

void rejects_independent_random_correspondences() {
    std::mt19937 random(31);
    robust_alignment::MatchResult matches;
    // Explicit engine scaling avoids distribution-library differences between platforms.
    const auto coordinate = [&random]() {
        return static_cast<float>(299.0 * static_cast<double>(random()) / std::mt19937::max());
    };
    for (int index = 0; index < 80; ++index) {
        matches.reference_points.emplace_back(coordinate(), coordinate());
        matches.moving_points.emplace_back(coordinate(), coordinate());
    }
    const auto result = robust_alignment::estimate_alignment(matches, cv::Size(300, 300), cv::Size(300, 300));
    require(!result.accepted && !result.reasons.empty(), "Unrelated random matches passed the acceptance gates");
    const cv::Mat image(300, 300, CV_8UC3, cv::Scalar(0));
    require_throws([&]() { robust_alignment::warp_alignment(image, result, image.size()); },
                   "Rejected geometry must not silently produce a trusted output", "rejected");
}

void rejects_inliers_covering_only_a_small_reference_patch() {
    robust_alignment::MatchResult matches;
    for (int y = 20; y < 45; y += 4) {
        for (int x = 20; x < 45; x += 4) {
            matches.reference_points.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    matches.moving_points = matches.reference_points;
    const auto result = robust_alignment::estimate_alignment(matches, cv::Size(500, 400), cv::Size(500, 400));
    require(!result.accepted, "A small repeated patch should not validate the entire reference");
    require(has_reason(result, "cover too little"), "Reference-support rejection did not explain its cause");
}

void rejects_a_projected_boundary_crossing_the_horizon() {
    robust_alignment::MatchResult matches;
    // All matched points lie on one finite side, while the complete reference crosses w=0.
    const cv::Mat forward(cv::Matx33d(3, 0, -400, 0, 1, 0, 0.01, 0, -1));
    for (int y = 30; y <= 150; y += 20) {
        for (int x = 200; x <= 350; x += 25) {
            matches.reference_points.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    matches.moving_points = robust_alignment::project_points(matches.reference_points, forward);
    const auto result = robust_alignment::estimate_alignment(matches, cv::Size(400, 320), cv::Size(400, 320));
    require(!result.accepted, "A target boundary crossing projective infinity was accepted");
    require(has_reason(result, "projective horizon"), "The projected horizon guard did not run: " + result.to_json());
}

void distinguishes_black_pixels_from_missing_source_support() {
    const cv::Mat black(50, 60, CV_8UC3, cv::Scalar(0));
    robust_alignment::AlignmentResult result;
    result.H = cv::Mat(cv::Matx33d(1, 0, 10, 0, 1, 0, 0, 0, 1));
    result.accepted = true;
    const auto warped = robust_alignment::warp_alignment(black, result, black.size());
    require(cv::countNonZero(warped.first.reshape(1)) == 0, "A black image changed color during the warp");
    require(warped.second.at<std::uint8_t>(25, 5) == 0, "The missing strip should have no source support");
    require(warped.second.at<std::uint8_t>(25, 20) == 255, "Real black content must remain valid");
}

void excludes_partial_bilinear_support_at_subpixel_boundaries() {
    cv::Mat white(18, 20, CV_8UC3, cv::Scalar::all(255));
    white(cv::Rect(8, 8, 4, 4)).setTo(cv::Scalar::all(0));
    robust_alignment::AlignmentResult result;
    result.H = cv::Mat(cv::Matx33d(1, 0, 0.4, 0, 1, 0.3, 0, 0, 1));
    result.accepted = true;
    const auto nearest = robust_alignment::warp_alignment(white, result, white.size());
    const auto full = robust_alignment::warp_alignment(white, result, white.size(), true, true);
    require(cv::norm(nearest.first, full.first, cv::NORM_INF) == 0, "Mask choice changed the aligned image");
    const auto edge_value = full.first.at<cv::Vec3b>(5, 0)[0];
    require(edge_value > 0 && edge_value < 255 && nearest.second.at<std::uint8_t>(5, 0) == 255,
            "Fixture must include a partially padded, nearest-valid edge");
    require(cv::countNonZero(full.second.col(0)) == 0 && cv::countNonZero(full.second.row(0)) == 0,
            "Full support retained partially padded edges");
    require(cv::countNonZero(full.second) == (white.cols - 1) * (white.rows - 1),
            "Full support excluded interior pixels");
    require(full.first.at<cv::Vec3b>(10, 10)[0] == 0 && full.second.at<std::uint8_t>(10, 10) == 255,
            "Fully supported black pixels must be valid");
    result.H = cv::Mat::eye(3, 3, CV_64F);
    const auto identity = robust_alignment::warp_alignment(white, result, white.size(), true, true);
    require(cv::countNonZero(identity.second) == white.cols * white.rows,
            "Zero-weight neighbors at the border must not invalidate exact pixel samples");
}

void prevents_unaccepted_warps() {
    const cv::Mat image(50, 60, CV_8UC3, cv::Scalar(0));
    robust_alignment::AlignmentResult rejected;
    rejected.H = cv::Mat::eye(3, 3, CV_64F);
    require_throws([&]() { robust_alignment::warp_alignment(image, rejected, image.size()); },
                   "The default warp should honor the rejection decision", "rejected");
    // Diagnostic overlays may deliberately inspect a rejected but finite model.
    const auto diagnostic = robust_alignment::warp_alignment(image, rejected, image.size(), false);
    require(cv::countNonZero(diagnostic.second) == image.rows * image.cols,
            "Explicit diagnostic warping did not retain the identity overlap");
    rejected.H.release();
    require_throws([&]() { robust_alignment::warp_alignment(image, rejected, image.size(), false); },
                   "A diagnostic request cannot replace a missing homography", "rejected");
}

void keeps_ecc_in_the_moving_to_reference_convention() {
    const cv::Mat reference = textured_image();
    const cv::Mat forward(cv::Matx33d(1.0, 0.01, 7, 0.02, 1.0, 3, 0.00001, -0.00001, 1));
    cv::Mat moving;
    cv::warpPerspective(reference, moving, forward, reference.size());
    cv::Mat initial = forward.inv();
    initial.at<double>(0, 2) += 1.5;
    initial.at<double>(1, 2) -= 1.0;
    const double before = corner_error(forward, initial, reference.size());
    const auto refined = robust_alignment::refine_ecc(reference, moving, initial);
    const double after = corner_error(forward, refined.first, reference.size());
    require(std::isfinite(refined.second), "ECC did not return a finite correlation");
    require(after < before && after < 0.75, "ECC inversion or initialization used the wrong coordinate direction");
}

void ecc_is_invariant_to_initial_projective_scale() {
    const cv::Mat reference = textured_image();
    const cv::Mat forward(cv::Matx33d(0.96, -0.02, 16, 0.03, 0.95, 8, 0.00022, -0.0001, 1));
    cv::Mat moving;
    cv::warpPerspective(reference, moving, forward, reference.size());
    cv::Mat initial = forward.inv();
    initial.at<double>(0, 2) += 0.7;
    initial.at<double>(1, 2) -= 0.5;
    const auto baseline = robust_alignment::refine_ecc(reference, moving, initial);
    require(corner_error(forward, baseline.first, reference.size()) < 0.5, "ECC failed to recover known corners");
    const auto corners = robust_alignment::image_corners(reference.size());
    const auto expected = robust_alignment::project_points(corners, baseline.first);
    for (const double scale : {8.0, -4.0, 1e-40, 1e40}) {
        const auto refined = robust_alignment::refine_ecc(reference, moving, initial * scale);
        const auto found = robust_alignment::project_points(corners, refined.first);
        for (std::size_t i = 0; i < corners.size(); ++i) {
            require(cv::norm(found[i] - expected[i]) < 1e-4, "ECC changed under equivalent projective scaling");
        }
        require(std::abs(refined.second - baseline.second) < 1e-6, "ECC score changed under projective scaling");
    }
}

void rejects_ecc_initialization_outside_its_parameterization() {
    const cv::Mat reference = textured_image();
    const cv::Mat swap_x_w(cv::Matx33d(0, 0, 1, 0, 1, 0, 1, 0, 0));
    require_throws([&]() { robust_alignment::refine_ecc(reference, reference, swap_x_w); },
                   "The inverse cannot be represented with a unit bottom-right entry", "cannot normalize");
}

void rejects_ecc_without_enough_overlap() {
    const cv::Mat reference = textured_image(cv::Size(100, 100));
    const cv::Mat no_overlap(cv::Matx33d(1, 0, -500, 0, 1, -500, 0, 0, 1));
    require_throws([&]() { robust_alignment::refine_ecc(reference, reference, no_overlap); },
                   "ECC should reject an initialization without valid source support", "Too little valid overlap");
    require_throws([&]() { robust_alignment::refine_ecc(reference, reference, cv::Mat::eye(3, 3, CV_64F), 0); },
                   "ECC should reject zero iterations", "positive");
}

void validates_correspondences_configuration_and_projection() {
    robust_alignment::MatchResult mismatch;
    mismatch.reference_points = {{0, 0}};
    require_throws([&]() { mismatch.validate(); }, "Mismatched point arrays should fail", "counts differ");
    mismatch.moving_points = {{0, 0}};
    mismatch.scores = {0.5F, 0.8F};
    require_throws([&]() { mismatch.validate(); }, "Mismatched score arrays should fail", "one score");
    mismatch.scores.clear();
    mismatch.reference_points.front().x = std::numeric_limits<float>::quiet_NaN();
    require_throws([&]() { mismatch.validate(); }, "Nonfinite coordinates should fail", "finite");
    robust_alignment::AlignmentConfig config;
    config.min_inliers = 3;
    require_throws([&]() { config.validate(); }, "A homography needs four inliers", "at least four");
    config = robust_alignment::AlignmentConfig{};
    config.min_reference_coverage = 1.1;
    require_throws([&]() { config.validate(); }, "Coverage outside [0,1] should fail", "[0, 1]");
    config = robust_alignment::AlignmentConfig{};
    config.reprojection_threshold = std::numeric_limits<double>::quiet_NaN();
    require_throws([&]() { config.validate(); }, "Nonfinite thresholds should fail", "finite");
    require_throws([]() { robust_alignment::image_corners(cv::Size(1, 20)); },
                   "Degenerate image dimensions should fail", "two pixels");
    const cv::Mat horizon(cv::Matx33d(1, 0, 0, 0, 1, 0, 1, 0, -5));
    require_throws([&]() { robust_alignment::project_points({{5, 0}}, horizon); },
                   "Projection at infinity should fail", "projective horizon");
    const cv::Mat perspective(cv::Matx33d(1, 0.1, 5, -0.02, 1, 8, 0.002, 0.01, 1));
    const auto expected = robust_alignment::project_points({{4, 7}}, perspective);
    for (const double factor : {-3.0, 1e-40, 1e40}) {
        require_throws([&]() { robust_alignment::project_points({{5, 0}}, factor * horizon); },
                       "Rescaling cannot move a point off the horizon", "projective horizon");
        const auto scaled = robust_alignment::project_points({{4, 7}}, factor * perspective);
        require(cv::norm(scaled.front() - expected.front()) < 1e-6,
                "Rescaling cannot move an ordinary projected point");
    }
    const cv::Mat scale(cv::Matx33d(2, 0, 4, 0, 3, -2, 0, 0, 1));
    const auto transformed = robust_alignment::project_points({{7, 9}}, scale);
    require(transformed.front() == cv::Point2f(18, 25), "Homogeneous projection changed x/y ordering");
}

void draws_empty_match_canvases_without_wrong_buffers() {
    const cv::Mat reference(50, 60, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat moving(70, 40, CV_8UC3, cv::Scalar(60, 70, 80));
    const robust_alignment::MatchResult empty;
    const cv::Mat canvas = robust_alignment::draw_matches(reference, moving, empty);
    require(canvas.size() == cv::Size(100, 70), "Match canvas dimensions do not contain both images");
    require(cv::norm(canvas(cv::Rect(0, 0, 60, 50)), reference, cv::NORM_INF) == 0,
            "The reference was copied into the wrong visualization buffer");
    require(cv::norm(canvas(cv::Rect(60, 0, 40, 70)), moving, cv::NORM_INF) == 0,
            "The moving image was copied into the wrong visualization buffer");
    require_throws([&]() { robust_alignment::draw_matches(reference, moving, empty, nullptr, 0); },
                   "A zero visualization limit should fail", "positive");
}

void serializes_missing_models_and_acceptance_evidence() {
    robust_alignment::AlignmentResult missing;
    missing.reasons = {"An explicit \"reason\"\nwith a newline"};
    const std::string text = missing.to_json();
    require(text.find("\"accepted\":false") != std::string::npos, "JSON booleans must not become strings");
    require(text.find("\"H\":null") != std::string::npos, "An absent homography must be JSON null");
    require(text.find("\\\"reason\\\"") != std::string::npos, "JSON strings did not escape quotation marks");
    require(text.find("\\u000a") != std::string::npos, "JSON strings did not escape control bytes");
    require(text.find("\"median_reprojection_error_px\":null") != std::string::npos,
            "Missing residual evidence must not be reported as a measured zero");
}

}  // namespace

int main() {
    cv::setNumThreads(1);  // Match the Python regressions' geometry execution policy.
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"known transform with outliers", recovers_known_transform_despite_outliers},
        {"small target in large scene", accepts_a_small_target_inside_a_large_scene},
        {"missing and collinear matches", rejects_missing_and_collinear_correspondences},
        {"independent random matches", rejects_independent_random_correspondences},
        {"reference coverage", rejects_inliers_covering_only_a_small_reference_patch},
        {"projective horizon", rejects_a_projected_boundary_crossing_the_horizon},
        {"black content and support mask", distinguishes_black_pixels_from_missing_source_support},
        {"full bilinear support", excludes_partial_bilinear_support_at_subpixel_boundaries},
        {"ECC projective scale equivalence", ecc_is_invariant_to_initial_projective_scale},
        {"ECC h22 parameterization", rejects_ecc_initialization_outside_its_parameterization},
        {"rejected warp contract", prevents_unaccepted_warps},
        {"ECC direction", keeps_ecc_in_the_moving_to_reference_convention},
        {"ECC overlap and arguments", rejects_ecc_without_enough_overlap},
        {"input and configuration contracts", validates_correspondences_configuration_and_projection},
        {"match visualization buffers", draws_empty_match_canvases_without_wrong_buffers},
        {"JSON evidence", serializes_missing_models_and_acceptance_evidence},
    };
    std::size_t passed = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            ++passed;
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "VALIDATION PASSED: " << passed << " native geometry tests; OpenCV " << CV_VERSION << '\n';
    return 0;
}
