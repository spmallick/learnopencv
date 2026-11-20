#include "alignment.hpp"

// OpenCV 5 moved homography estimation into the geometry module.
#include <opencv2/geometry/3d.hpp>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace robust_alignment {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check_image(const cv::Mat& image) {
    // A single explicit contract prevents uint8 BGR and normalized RGB tensor confusion.
    if (image.empty() || image.type() != CV_8UC3 || image.rows < 2 || image.cols < 2) {
        throw std::invalid_argument("Expected a nonempty uint8 BGR image at least 2 by 2 pixels.");
    }
}

cv::Mat checked_homography(const cv::Mat& H) {
    // Work in double precision even if a caller supplied a float ECC matrix.
    if (H.rows != 3 || H.cols != 3 || H.channels() != 1) {
        throw std::invalid_argument("Expected a finite 3-by-3 homography.");
    }
    cv::Mat result;
    H.convertTo(result, CV_64F);
    if (!cv::checkRange(result)) {
        throw std::invalid_argument("Expected a finite 3-by-3 homography.");
    }
    return result;
}

cv::Mat inverse_homography(const cv::Mat& H) {
    cv::Mat inverse;
    // DECOMP_LU reports a singular matrix instead of silently supplying a pseudoinverse.
    if (cv::invert(checked_homography(H), inverse, cv::DECOMP_LU) == 0.0) {
        throw std::invalid_argument("Singular matrix");
    }
    return inverse;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("A median requires at least one value.");
    }
    // Sorting makes the even-count convention explicit and matches NumPy's median.
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

void write_matrix_array(cv::FileStorage& storage, const std::string& key, const cv::Mat& H) {
    storage << key << "[";
    if (!H.empty()) {
        const cv::Mat matrix = checked_homography(H);
        for (int row = 0; row < 3; ++row) {
            storage << "[";
            for (int column = 0; column < 3; ++column) {
                storage << matrix.at<double>(row, column);
            }
            storage << "]";
        }
    }
    storage << "]";
}

std::string quote_json(const std::string& text) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : text) {
        // JSON requires escaping control bytes and the two structural string characters.
        if (character == '"' || character == '\\') {
            output << '\\' << character;
        } else if (character < 0x20) {
            output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<unsigned int>(character) << std::dec;
        } else {
            output << character;
        }
    }
    output << '"';
    return output.str();
}

cv::Point rounded_point(const cv::Point2f& point) {
    // nearbyint uses nearest-even rounding, as NumPy rint does in the Python visualization.
    const double x = std::nearbyint(static_cast<double>(point.x));
    const double y = std::nearbyint(static_cast<double>(point.y));
    const double lower = static_cast<double>(std::numeric_limits<int>::min());
    const double upper = static_cast<double>(std::numeric_limits<int>::max());
    if (x < lower || x > upper || y < lower || y > upper) {
        throw std::invalid_argument("A match coordinate exceeds the drawable integer range.");
    }
    return cv::Point(static_cast<int>(x), static_cast<int>(y));
}

}  // namespace

void MatchResult::validate() const {
    if (reference_points.size() != moving_points.size()) {
        throw std::invalid_argument("Reference and moving point counts differ.");
    }
    if (!scores.empty() && scores.size() != reference_points.size()) {
        throw std::invalid_argument("There must be one score per correspondence, or no scores.");
    }
    for (const auto* points : {&reference_points, &moving_points}) {
        for (const cv::Point2f& point : *points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                throw std::invalid_argument("Correspondences must contain only finite coordinates.");
            }
        }
    }
}

void AlignmentConfig::validate() const {
    if (!std::isfinite(reprojection_threshold) || !std::isfinite(max_median_reprojection_error)
        || reprojection_threshold <= 0.0 || max_median_reprojection_error <= 0.0) {
        throw std::invalid_argument("Pixel error thresholds must be positive and finite.");
    }
    if (!(confidence > 0.0 && confidence < 1.0) || max_iterations < 1 || min_inliers < 4) {
        throw std::invalid_argument("Use 0 < confidence < 1, positive iterations, and at least four inliers.");
    }
    for (const double value : {min_inlier_ratio, min_reference_coverage, min_visible_fraction}) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            throw std::invalid_argument("Ratios and coverage thresholds must lie in [0, 1].");
        }
    }
    if (!std::isfinite(min_projected_area) || min_projected_area < 0.0) {
        throw std::invalid_argument("min_projected_area must be finite and nonnegative.");
    }
}

void AlignmentConfig::write(cv::FileStorage& storage) const {
    storage << "reprojection_threshold" << reprojection_threshold << "confidence" << confidence;
    storage << "max_iterations" << max_iterations << "min_inliers" << min_inliers;
    storage << "min_inlier_ratio" << min_inlier_ratio << "min_reference_coverage" << min_reference_coverage;
    storage << "max_median_reprojection_error" << max_median_reprojection_error;
    storage << "min_projected_area" << min_projected_area << "min_visible_fraction" << min_visible_fraction;
}

std::size_t AlignmentResult::num_inliers() const {
    return static_cast<std::size_t>(std::count_if(inliers.begin(), inliers.end(),
                                                [](std::uint8_t value) { return value != 0; }));
}

double AlignmentResult::inlier_ratio() const {
    return static_cast<double>(num_inliers()) / static_cast<double>(std::max<std::size_t>(num_matches, 1));
}

void AlignmentResult::write(cv::FileStorage& storage) const {
    storage << "accepted" << static_cast<int>(accepted) << "reasons" << "[";
    for (const std::string& reason : reasons) {
        storage << reason;
    }
    storage << "]" << "homography_direction" << "moving_to_reference";
    write_matrix_array(storage, "H", H);
    storage << "num_matches" << static_cast<int>(num_matches) << "num_inliers" << static_cast<int>(num_inliers());
    storage << "inlier_ratio" << inlier_ratio();
    if (median_reprojection_error) {
        storage << "median_reprojection_error_px" << *median_reprojection_error;
    }
    storage << "reference_inlier_hull_coverage" << reference_coverage;
    storage << "moving_inlier_hull_coverage" << moving_coverage;
    storage << "projected_reference_visible_fraction" << visible_fraction;
    storage << "projected_reference_quad" << "[";
    for (const cv::Point2f& point : projected_reference_quad) {
        storage << "[" << point.x << point.y << "]";
    }
    storage << "]" << "estimator" << estimator << "estimation_ms" << estimation_ms;
}

std::string AlignmentResult::to_json() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());  // A decimal comma would make invalid JSON.
    output << std::setprecision(17) << "{\"accepted\":" << (accepted ? "true" : "false") << ",\"reasons\":[";
    for (std::size_t index = 0; index < reasons.size(); ++index) {
        output << (index == 0 ? "" : ",") << quote_json(reasons[index]);
    }
    output << "],\"homography_direction\":\"moving_to_reference\",\"H\":";
    if (H.empty()) {
        output << "null";
    } else {
        const cv::Mat matrix = checked_homography(H);
        output << '[';
        for (int row = 0; row < 3; ++row) {
            output << (row == 0 ? "[" : ",[");
            for (int column = 0; column < 3; ++column) {
                output << (column == 0 ? "" : ",") << matrix.at<double>(row, column);
            }
            output << ']';
        }
        output << ']';
    }
    output << ",\"num_matches\":" << num_matches << ",\"num_inliers\":" << num_inliers();
    output << ",\"inlier_ratio\":" << inlier_ratio() << ",\"median_reprojection_error_px\":";
    if (median_reprojection_error) {
        output << *median_reprojection_error;
    } else {
        output << "null";
    }
    output << ",\"reference_inlier_hull_coverage\":" << reference_coverage;
    output << ",\"moving_inlier_hull_coverage\":" << moving_coverage;
    output << ",\"projected_reference_visible_fraction\":" << visible_fraction;
    output << ",\"projected_reference_quad\":";
    if (projected_reference_quad.empty()) {
        output << "null";
    } else {
        output << '[';
        for (std::size_t index = 0; index < projected_reference_quad.size(); ++index) {
            const cv::Point2f& point = projected_reference_quad[index];
            output << (index == 0 ? "[" : ",[") << point.x << ',' << point.y << ']';
        }
        output << ']';
    }
    output << ",\"estimator\":" << quote_json(estimator) << ",\"estimation_ms\":" << estimation_ms << '}';
    return output.str();
}

cv::Mat load_image(const std::filesystem::path& path) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Cannot read image: " + path.string());
    }
    return image;
}

std::vector<cv::Point2f> image_corners(cv::Size image_size) {
    if (image_size.width < 2 || image_size.height < 2) {
        throw std::invalid_argument("Image dimensions must be at least two pixels.");
    }
    // The outermost pixel centers use width-1 and height-1, not image boundary coordinates.
    const float right = static_cast<float>(image_size.width - 1);
    const float bottom = static_cast<float>(image_size.height - 1);
    return {{0.0F, 0.0F}, {right, 0.0F}, {right, bottom}, {0.0F, bottom}};
}

std::vector<cv::Point2f> project_points(const std::vector<cv::Point2f>& points, const cv::Mat& H) {
    const cv::Mat matrix = checked_homography(H);
    std::vector<cv::Point2f> projected;
    projected.reserve(points.size());
    for (const cv::Point2f& point : points) {
        const double x = static_cast<double>(point.x);
        const double y = static_cast<double>(point.y);
        const double wx = matrix.at<double>(2, 0) * x;
        const double wy = matrix.at<double>(2, 1) * y;
        const double constant = matrix.at<double>(2, 2);
        const double w = wx + wy + constant;
        // A relative cancellation check gives equivalent projective scales the same decision.
        const double denominator_scale = std::abs(wx) + std::abs(wy) + std::abs(constant);
        if (!std::isfinite(w) || std::abs(w) <= 1e-10 * denominator_scale) {
            throw std::invalid_argument("A projected point lies on the projective horizon.");
        }
        // Explicit homogeneous division makes the source/destination convention inspectable.
        const double u = (matrix.at<double>(0, 0) * x + matrix.at<double>(0, 1) * y + matrix.at<double>(0, 2)) / w;
        const double v = (matrix.at<double>(1, 0) * x + matrix.at<double>(1, 1) * y + matrix.at<double>(1, 2)) / w;
        const cv::Point2f result(static_cast<float>(u), static_cast<float>(v));
        if (!std::isfinite(result.x) || !std::isfinite(result.y)) {
            throw std::invalid_argument("Homography projection produced nonfinite coordinates.");
        }
        projected.push_back(result);
    }
    return projected;
}

double hull_coverage(const std::vector<cv::Point2f>& points, cv::Size image_size) {
    image_corners(image_size);  // Validate dimensions even when there are too few points.
    if (points.size() < 3) {
        return 0.0;
    }
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    const double image_area = static_cast<double>(image_size.width - 1) * (image_size.height - 1);
    return cv::contourArea(hull) / image_area;
}

AlignmentResult estimate_alignment(const MatchResult& matches, cv::Size reference_size,
                                   cv::Size moving_size, const AlignmentConfig& config,
                                   const std::string& estimator) {
    matches.validate();
    config.validate();
    if (estimator != "magsac" && estimator != "ransac") {
        throw std::invalid_argument("estimator must be magsac or ransac.");
    }
    AlignmentResult result;
    result.num_matches = matches.reference_points.size();
    result.inliers.assign(result.num_matches, 0);
    result.estimator = estimator;
    const auto reference_corners = image_corners(reference_size);
    const auto moving_corners = image_corners(moving_size);
    if (result.num_matches < 4) {
        result.reasons.emplace_back("Fewer than four correspondences.");
        return result;
    }
    if (hull_coverage(matches.reference_points, reference_size) < 1e-6
        || hull_coverage(matches.moving_points, moving_size) < 1e-6) {
        result.reasons.emplace_back("Correspondences are collinear or have negligible spatial support.");
        return result;
    }
    cv::Mat mask;
    const auto start = Clock::now();
    try {
        // Source first and destination second: H maps moving-image points into the reference.
        result.H = cv::findHomography(matches.moving_points, matches.reference_points,
                                     estimator == "magsac" ? cv::USAC_MAGSAC : cv::RANSAC,
                                     config.reprojection_threshold, mask, config.max_iterations, config.confidence);
    } catch (const cv::Exception& error) {
        result.estimation_ms = elapsed_ms(start);
        result.reasons.emplace_back("Homography estimation failed: " + error.err);
        return result;
    }
    result.estimation_ms = elapsed_ms(start);
    if (result.H.empty() || mask.empty() || !cv::checkRange(result.H)) {
        result.H.release();
        result.reasons.emplace_back("No finite homography was estimated.");
        return result;
    }
    result.H.convertTo(result.H, CV_64F);
    const double scale = std::abs(result.H.at<double>(2, 2)) > 1e-12
                             ? result.H.at<double>(2, 2) : cv::norm(result.H);
    result.H /= scale;  // Scale-equivalent matrices receive a readable normalization.
    const cv::Mat flat_mask = mask.reshape(1, 1);
    for (std::size_t index = 0; index < result.num_matches; ++index) {
        result.inliers[index] = flat_mask.at<std::uint8_t>(0, static_cast<int>(index)) != 0 ? 1 : 0;
    }
    if (result.num_inliers() < static_cast<std::size_t>(config.min_inliers)) {
        result.reasons.emplace_back("Only " + std::to_string(result.num_inliers())
                                    + " inliers; need " + std::to_string(config.min_inliers) + ".");
    }
    if (result.inlier_ratio() < config.min_inlier_ratio) {
        result.reasons.emplace_back("Inlier ratio is below the configured minimum.");
    }
    std::vector<cv::Point2f> reference_inliers;
    std::vector<cv::Point2f> moving_inliers;
    for (std::size_t index = 0; index < result.num_matches; ++index) {
        if (result.inliers[index] != 0) {
            reference_inliers.push_back(matches.reference_points[index]);
            moving_inliers.push_back(matches.moving_points[index]);
        }
    }
    result.reference_coverage = hull_coverage(reference_inliers, reference_size);
    result.moving_coverage = hull_coverage(moving_inliers, moving_size);
    if (result.reference_coverage < config.min_reference_coverage) {
        result.reasons.emplace_back("Inliers cover too little of the reference image.");
    }
    try {
        if (!reference_inliers.empty()) {
            const auto projected = project_points(moving_inliers, result.H);
            std::vector<double> errors;
            errors.reserve(projected.size());
            for (std::size_t index = 0; index < projected.size(); ++index) {
                errors.push_back(cv::norm(projected[index] - reference_inliers[index]));
            }
            result.median_reprojection_error = median(std::move(errors));
            if (*result.median_reprojection_error > config.max_median_reprojection_error) {
                result.reasons.emplace_back("Median inlier reprojection error exceeds the configured limit.");
            }
        }
        const cv::Mat inverse = inverse_homography(result.H);
        double minimum_denominator = std::numeric_limits<double>::infinity();
        double maximum_denominator = -std::numeric_limits<double>::infinity();
        for (const cv::Point2f& corner : reference_corners) {
            const double denominator = inverse.at<double>(2, 0) * corner.x
                                       + inverse.at<double>(2, 1) * corner.y + inverse.at<double>(2, 2);
            minimum_denominator = std::min(minimum_denominator, denominator);
            maximum_denominator = std::max(maximum_denominator, denominator);
        }
        if (minimum_denominator < 0.0 && maximum_denominator > 0.0) {
            throw std::invalid_argument("The projected reference crosses the projective horizon.");
        }
        result.projected_reference_quad = project_points(reference_corners, inverse);
        const double area = std::abs(cv::contourArea(result.projected_reference_quad));
        if (!cv::isContourConvex(result.projected_reference_quad) || area < config.min_projected_area) {
            result.reasons.emplace_back("Projected reference boundary is degenerate or too small.");
        } else {
            std::vector<cv::Point2f> intersection;
            const float intersection_area = cv::intersectConvexConvex(result.projected_reference_quad,
                                                                     moving_corners, intersection);
            result.visible_fraction = std::clamp(static_cast<double>(intersection_area) / area, 0.0, 1.0);
            if (result.visible_fraction < config.min_visible_fraction) {
                result.reasons.emplace_back("Too little of the projected reference lies inside the moving image.");
            }
        }
    } catch (const std::exception& error) {
        result.reasons.emplace_back(std::string("Invalid projected geometry: ") + error.what());
    }
    result.accepted = result.reasons.empty();
    return result;
}

std::pair<cv::Mat, cv::Mat> warp_alignment(const cv::Mat& moving_bgr, const AlignmentResult& result,
                                        cv::Size reference_size, bool require_accepted, bool full_support) {
    check_image(moving_bgr);
    image_corners(reference_size);
    if (result.H.empty() || (require_accepted && !result.accepted)) {
        throw std::invalid_argument("Alignment was rejected; inspect the report before warping.");
    }
    cv::Mat aligned;
    cv::warpPerspective(moving_bgr, aligned, result.H, reference_size, cv::INTER_LINEAR);
    cv::Mat valid;
    if (full_support) {
        // A float mask preserves fractional weights, including tiny corner contributions.
        const cv::Mat source_ones(moving_bgr.size(), CV_32F, cv::Scalar(1));
        cv::Mat weight;
        cv::warpPerspective(source_ones, weight, result.H, reference_size, cv::INTER_LINEAR);
        cv::compare(weight, 1.0, valid, cv::CMP_EQ);
    } else {
        // Nearest-source validity is independent of the image colors, including black.
        const cv::Mat source_mask(moving_bgr.size(), CV_8UC1, cv::Scalar(255));
        cv::warpPerspective(source_mask, valid, result.H, reference_size, cv::INTER_NEAREST);
    }
    return {aligned, valid};
}

std::pair<cv::Mat, double> refine_ecc(const cv::Mat& reference_bgr, const cv::Mat& moving_bgr,
                                   const cv::Mat& H_moving_to_reference, int iterations, double epsilon) {
    check_image(reference_bgr);
    check_image(moving_bgr);
    if (iterations < 1 || !std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("ECC iterations and epsilon must be positive.");
    }
    cv::Mat reference_gray;
    cv::Mat moving_gray;
    cv::cvtColor(reference_bgr, reference_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(moving_bgr, moving_gray, cv::COLOR_BGR2GRAY);
    reference_gray.convertTo(reference_gray, CV_32F, 1.0 / 255.0);
    moving_gray.convertTo(moving_gray, CV_32F, 1.0 / 255.0);
    // Remove arbitrary projective scale before inversion to handle equivalent inputs alike.
    const cv::Mat initial = checked_homography(H_moving_to_reference);
    const double magnitude = cv::norm(initial, cv::NORM_INF);
    if (magnitude == 0.0) {
        throw std::invalid_argument("ECC needs a nonzero initial homography.");
    }
    const cv::Mat inverse = inverse_homography(initial / magnitude);
    // The ECC Jacobian fixes h22 to one in template-to-input coordinates. Inverting
    // generally changes this entry even when the public moving-to-reference H had h22=1.
    const double h22 = inverse.at<double>(2, 2);
    if (!cv::checkRange(inverse) || std::abs(h22) <= 1e-12 * cv::norm(inverse, cv::NORM_INF)) {
        throw std::invalid_argument("ECC cannot normalize its reference-to-moving initialization at [2, 2].");
    }
    cv::Mat reference_to_moving;
    inverse.convertTo(reference_to_moving, CV_32F, 1.0 / h22);
    const cv::Mat reference_support(reference_bgr.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat input_mask;
    cv::warpPerspective(reference_support, input_mask, reference_to_moving, moving_bgr.size(), cv::INTER_NEAREST);
    // Remove Gaussian support near the estimated overlap boundary before optimizing intensities.
    cv::erode(input_mask, input_mask, cv::Mat::ones(7, 7, CV_8U), cv::Point(-1, -1), 1,
              cv::BORDER_CONSTANT, cv::Scalar(0));
    if (cv::countNonZero(input_mask) < 64) {
        throw std::invalid_argument("Too little valid overlap for ECC refinement.");
    }
    const cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, iterations, epsilon);
    const double correlation = cv::findTransformECC(reference_gray, moving_gray, reference_to_moving,
                                                    cv::MOTION_HOMOGRAPHY, criteria, input_mask, 5);
    // Return to the public direction in double precision. The inverse can have h22=0.
    cv::Mat H = inverse_homography(reference_to_moving);
    if (!cv::checkRange(H) || !std::isfinite(correlation)) {
        throw std::invalid_argument("ECC returned a nonfinite refinement.");
    }
    const double max_entry = cv::norm(H, cv::NORM_INF);
    const double scale = H.at<double>(2, 2);
    H /= std::abs(scale) > 1e-12 * max_entry ? scale : max_entry;
    return {H, correlation};
}

cv::Mat draw_matches(const cv::Mat& reference_bgr, const cv::Mat& moving_bgr, const MatchResult& matches,
                     const AlignmentResult* result, int max_draw) {
    check_image(reference_bgr);
    check_image(moving_bgr);
    matches.validate();
    if (max_draw < 1 || (result != nullptr && result->inliers.size() != matches.reference_points.size())) {
        throw std::invalid_argument("Use a positive match limit and an inlier mask matching the point count.");
    }
    const int offset = reference_bgr.cols;
    cv::Mat canvas(std::max(reference_bgr.rows, moving_bgr.rows), offset + moving_bgr.cols,
                   CV_8UC3, cv::Scalar(0, 0, 0));
    reference_bgr.copyTo(canvas(cv::Rect(0, 0, reference_bgr.cols, reference_bgr.rows)));
    moving_bgr.copyTo(canvas(cv::Rect(offset, 0, moving_bgr.cols, moving_bgr.rows)));
    const std::size_t count = std::min(static_cast<std::size_t>(max_draw), matches.reference_points.size());
    for (std::size_t drawn = 0; drawn < count; ++drawn) {
        // The same evenly spaced subset as np.linspace(..., dtype=int) keeps plots reproducible.
        const std::size_t index = count == 1 ? 0 : static_cast<std::size_t>(
            static_cast<double>(drawn) * static_cast<double>(matches.reference_points.size() - 1)
            / static_cast<double>(count - 1));
        const cv::Point point0 = rounded_point(matches.reference_points[index]);
        const cv::Point point1 = rounded_point(matches.moving_points[index] + cv::Point2f(static_cast<float>(offset), 0));
        const cv::Scalar color = result == nullptr || result->inliers[index] != 0
                                     ? cv::Scalar(40, 200, 40) : cv::Scalar(40, 40, 220);
        cv::line(canvas, point0, point1, color, 1, cv::LINE_AA);
        cv::circle(canvas, point0, 3, color, -1, cv::LINE_AA);
        cv::circle(canvas, point1, 3, color, -1, cv::LINE_AA);
    }
    return canvas;
}

}  // namespace robust_alignment
