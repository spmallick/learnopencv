#pragma once

// OpenCV core supplies matrices, points, norms, ranges, and filesystem globs.
#include <opencv2/core.hpp>
// The version macro selects headers from the OpenCV 4 or OpenCV 5 layout.
#include <opencv2/core/version.hpp>
// HighGUI is used only when the reader keeps interactive display enabled.
#include <opencv2/highgui.hpp>
// Image codecs load checkerboards and save both undistorted outputs.
#include <opencv2/imgcodecs.hpp>
// Image processing supplies color conversion, corner refinement, and remapping.
#include <opencv2/imgproc.hpp>

// OpenCV 5 split calibration and geometry out of the old calib3d module and
// moved checkerboard detection into objdetect.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib.hpp>
#include <opencv2/geometry/3d.hpp>
#include <opencv2/objdetect.hpp>
#else
// OpenCV 4 keeps all calibration APIs used here in calib3d.
#include <opencv2/calib3d.hpp>
#endif

// Sorting the glob results makes the default sample deterministic.
#include <algorithm>
// Square root, finite checks, and absolute differences validate numeric output.
#include <cmath>
// Filesystem paths provide portable input and output handling.
#include <filesystem>
// Precision control keeps reported errors comparable with the Python lesson.
#include <iomanip>
// The tutorial prints calibration results and validation markers.
#include <iostream>
// Runtime errors turn invalid input and output into clear CLI failures.
#include <stdexcept>
// Strings carry image patterns and readable diagnostics.
#include <string>
// Vectors store one point set and one pose for each checkerboard view.
#include <vector>

// Direct compilation can override this macro; CMake supplies an absolute path.
#ifndef TUTORIAL_IMAGE_DIR
#define TUTORIAL_IMAGE_DIR "./images"
#endif

namespace tutorial {

// Keep verbose standard-library qualification out of path-heavy code only.
namespace fs = std::filesystem;

// The printed checkerboard contains six inner corners across and nine down.
inline const cv::Size checkerboard_size{6, 9};
// Refine corners for at most 30 iterations or until 0.001-pixel convergence.
inline const cv::TermCriteria corner_criteria{
    cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
    30,
    0.001,
};
// These version-compatible flags normalize lighting, adapt thresholding, and
// quickly reject images that clearly cannot contain the requested board.
inline const int chessboard_flags =
    cv::CALIB_CB_ADAPTIVE_THRESH |
    cv::CALIB_CB_FAST_CHECK |
    cv::CALIB_CB_NORMALIZE_IMAGE;

// Group the camera model, view poses, errors, and coverage in one named value.
struct CalibrationResult {
    // OpenCV's nonlinear optimizer returns its RMS reprojection error.
    double rms_error = 0.0;
    // The tutorial independently recomputes RMSE from projected points.
    double reprojection_rmse = 0.0;
    // This 3-by-3 matrix stores focal lengths and the optical center.
    cv::Mat camera_matrix;
    // These coefficients model radial and tangential lens distortion.
    cv::Mat distortion_coefficients;
    // Each detected view receives one Rodrigues rotation vector.
    std::vector<cv::Mat> rotation_vectors;
    // Each detected view receives one translation vector.
    std::vector<cv::Mat> translation_vectors;
    // OpenCV stores image size as width followed by height.
    cv::Size image_size;
    // The full match list lets tests detect missing fixture files.
    std::vector<cv::String> image_paths;
    // This count can be lower than the match count for undetected boards.
    std::size_t detected_images = 0;
};

// Detect checkerboards and estimate a camera model from every matching image.
inline CalibrationResult calibrate_images(
    const std::string& image_pattern,
    const bool display
) {
    // Start with an empty result that the function fills consistently.
    CalibrationResult result;
    // OpenCV's glob keeps wildcard expansion portable across shells.
    cv::glob(image_pattern, result.image_paths, false);
    // Explicit sorting makes the first sample stable on every filesystem.
    std::sort(result.image_paths.begin(), result.image_paths.end());
    // Prevent calibration from receiving an empty observation set.
    if (result.image_paths.empty()) {
        throw std::runtime_error(
            "No calibration images matched: " + image_pattern
        );
    }

    // Model the planar board with one world unit between adjacent corners.
    std::vector<cv::Point3f> object_template;
    // Reserve exactly one slot for every checkerboard corner.
    object_template.reserve(
        static_cast<std::size_t>(checkerboard_size.area())
    );
    // Match the row-major order returned by checkerboard detection.
    for (int row = 0; row < checkerboard_size.height; ++row) {
        for (int column = 0; column < checkerboard_size.width; ++column) {
            // All known points lie on the z=0 checkerboard plane.
            object_template.emplace_back(
                static_cast<float>(column),
                static_cast<float>(row),
                0.0F
            );
        }
    }

    // Repeat the known 3D grid for every successful view.
    std::vector<std::vector<cv::Point3f>> object_points;
    // Store the matching sub-pixel image coordinates for every view.
    std::vector<std::vector<cv::Point2f>> image_points;

    try {
        // Process all matches so unreadable or inconsistent assets fail clearly.
        for (const cv::String& image_path : result.image_paths) {
            // Preserve a color image for the optional teaching overlay.
            cv::Mat image = cv::imread(image_path);
            // imread returns an empty matrix rather than throwing on failure.
            if (image.empty()) {
                throw std::runtime_error(
                    "Could not read calibration image: " + image_path
                );
            }

            // Checkerboard detection needs a single intensity channel.
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            // The first readable image establishes the calibration resolution.
            const cv::Size current_size = gray.size();
            if (result.image_size.empty()) {
                result.image_size = current_size;
            } else if (current_size != result.image_size) {
                // Intrinsics cannot be estimated from unscaled mixed resolutions.
                throw std::runtime_error(
                    "All calibration images must have the same size; " +
                    image_path + " differs from the first image"
                );
            }

            // Search for the requested six-by-nine inner-corner grid.
            std::vector<cv::Point2f> corners;
            const bool found = cv::findChessboardCorners(
                gray,
                checkerboard_size,
                corners,
                chessboard_flags
            );
            // Only a complete board contributes correspondences.
            if (found) {
                // Improve detector coordinates to sub-pixel accuracy.
                cv::cornerSubPix(
                    gray,
                    corners,
                    cv::Size(11, 11),
                    cv::Size(-1, -1),
                    corner_criteria
                );
                // Vector copying gives this observation its own 3D grid.
                object_points.push_back(object_template);
                // Preserve the refined 2D coordinates for optimization.
                image_points.push_back(corners);
                // Draw only for the interactive explanation; calibration uses
                // the unmodified grayscale pixels and stored coordinates.
                cv::drawChessboardCorners(
                    image,
                    checkerboard_size,
                    corners,
                    found
                );
            }

            // Headless validation avoids all window-system operations.
            if (display) {
                cv::imshow("Checkerboard corners", image);
                // Pause after each view so the reader can inspect the overlay.
                cv::waitKey(0);
            }
        }
    } catch (...) {
        // Close a window opened by an earlier image before propagating failure.
        if (display) {
            cv::destroyAllWindows();
        }
        throw;
    }
    // Release GUI resources after the final successful preview.
    if (display) {
        cv::destroyAllWindows();
    }
    // Three views are a practical minimum for this tutorial's camera model.
    if (object_points.size() < 3) {
        throw std::runtime_error(
            "Calibration needs at least 3 valid checkerboards; found " +
            std::to_string(object_points.size())
        );
    }

    // Optimize intrinsics, distortion, and one board pose per detected image.
    result.rms_error = cv::calibrateCamera(
        object_points,
        image_points,
        result.image_size,
        result.camera_matrix,
        result.distortion_coefficients,
        result.rotation_vectors,
        result.translation_vectors
    );
    // Preserve coverage independently of the complete input match count.
    result.detected_images = object_points.size();

    // Accumulate squared x/y residuals in double precision.
    double squared_error = 0.0;
    // Divide by observed points, not by images, when computing RMSE.
    std::size_t point_count = 0;
    // Reproject each known grid with its optimized camera and pose.
    for (std::size_t index = 0; index < object_points.size(); ++index) {
        // projectPoints returns one image coordinate per known world point.
        std::vector<cv::Point2f> projected;
        cv::projectPoints(
            object_points[index],
            result.rotation_vectors[index],
            result.translation_vectors[index],
            result.camera_matrix,
            result.distortion_coefficients,
            projected
        );
        // The L2 norm combines residuals across both coordinate axes.
        const double l2_error = cv::norm(
            image_points[index],
            projected,
            cv::NORM_L2
        );
        // Squaring the norm recovers the sum of squared residual components.
        squared_error += l2_error * l2_error;
        // Every grid point has one projected and one observed coordinate.
        point_count += object_points[index].size();
    }
    // Convert mean squared point error back to pixels.
    result.reprojection_rmse = std::sqrt(
        squared_error / static_cast<double>(point_count)
    );
    // Return all reusable calibration values and testable metadata.
    return result;
}

// Print the coverage and reusable calibration products for tutorial readers.
inline void print_calibration(const CalibrationResult& result) {
    // Begin with input coverage so a plausible matrix cannot hide missing views.
    std::cout << "images=" << result.image_paths.size()
              << ", detections=" << result.detected_images
              << ", image_size=" << result.image_size << '\n';
    // Preserve enough digits to compare the C++ and Python implementations.
    std::cout << std::setprecision(9);
    // Report OpenCV's optimizer error and the independently calculated error.
    std::cout << "RMS calibration error: "
              << result.rms_error << " pixels\n";
    std::cout << "Reprojection RMSE: "
              << result.reprojection_rmse << " pixels\n";
    // Intrinsics and distortion define the reusable camera model.
    std::cout << "Camera matrix:\n" << result.camera_matrix << '\n';
    std::cout << "Distortion coefficients:\n"
              << result.distortion_coefficients << '\n';
    // Preserve the original tutorial's per-view rotation and translation output.
    std::cout << "Rotation vectors:\n";
    for (std::size_t index = 0;
         index < result.rotation_vectors.size();
         ++index) {
        std::cout << '[' << index << "]\n"
                  << result.rotation_vectors[index] << '\n';
    }
    std::cout << "Translation vectors:\n";
    for (std::size_t index = 0;
         index < result.translation_vectors.size();
         ++index) {
        std::cout << '[' << index << "]\n"
                  << result.translation_vectors[index] << '\n';
    }
}

// Check version-independent invariants before a result is trusted or published.
inline void validate_calibration(const CalibrationResult& result) {
    // Guard the minimum even when validating a stored result.
    if (result.detected_images < 3) {
        throw std::runtime_error(
            "At least three checkerboards must be detected"
        );
    }
    // Every detected view must have one rotation and translation estimate.
    if (result.rotation_vectors.size() != result.detected_images ||
        result.translation_vectors.size() != result.detected_images) {
        throw std::runtime_error(
            "Calibration pose counts do not match the detections"
        );
    }
    // Intrinsics must use OpenCV's normalized 3-by-3 double matrix form.
    if (result.camera_matrix.rows != 3 ||
        result.camera_matrix.cols != 3 ||
        result.camera_matrix.type() != CV_64F ||
        !cv::checkRange(result.camera_matrix) ||
        std::abs(result.camera_matrix.at<double>(2, 2) - 1.0) >= 1e-12) {
        throw std::runtime_error(
            "Camera matrix has an invalid shape, type, range, or scale"
        );
    }
    // Positive focal lengths are required by a physically meaningful model.
    if (result.camera_matrix.at<double>(0, 0) <= 0.0 ||
        result.camera_matrix.at<double>(1, 1) <= 0.0) {
        throw std::runtime_error("Focal lengths must be positive");
    }
    // Distortion terms must remain finite before map construction.
    if (result.distortion_coefficients.empty() ||
        !cv::checkRange(result.distortion_coefficients)) {
        throw std::runtime_error(
            "Distortion coefficients are empty or non-finite"
        );
    }
    // Both reported errors should be finite and below one pixel for this lesson.
    if (!std::isfinite(result.rms_error) || result.rms_error >= 1.0) {
        throw std::runtime_error(
            "Expected calibration RMS below 1 pixel"
        );
    }
    if (!std::isfinite(result.reprojection_rmse) ||
        result.reprojection_rmse >= 1.0) {
        throw std::runtime_error(
            "Expected reprojection RMSE below 1 pixel"
        );
    }
    // Both values summarize the same residuals and differ only by rounding.
    if (std::abs(result.rms_error - result.reprojection_rmse) >= 1e-5) {
        throw std::runtime_error(
            "OpenCV RMS and independently calculated reprojection RMSE disagree"
        );
    }
}

// Keep both images and their scalar comparison together.
struct UndistortionResult {
    // Direct undistort computes and applies its mapping in one call.
    cv::Mat direct;
    // remap applies a separately precomputed floating-point map.
    cv::Mat remapped;
    // Mean absolute pixel difference quantifies method agreement.
    double mean_absolute_difference = 0.0;
    // Preserve the two exact destinations for validation and diagnostics.
    fs::path direct_path;
    fs::path remap_path;
};

// Undistort one same-resolution image through both documented workflows.
inline UndistortionResult undistort_sample(
    const CalibrationResult& calibration,
    const fs::path& sample_path,
    const fs::path& output_dir,
    const bool display
) {
    // Load color pixels for output and side-by-side visualization.
    const cv::Mat image = cv::imread(sample_path.string());
    // imread signals path and decode failures with an empty matrix.
    if (image.empty()) {
        throw std::runtime_error(
            "Could not read image to undistort: " + sample_path.string()
        );
    }
    // Reusing intrinsics at another resolution requires explicit rescaling.
    if (image.size() != calibration.image_size) {
        throw std::runtime_error(
            "Undistortion image size differs from calibration images"
        );
    }

    // Alpha=1 retains all source pixels, including invalid border regions.
    const cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        calibration.image_size,
        1.0,
        calibration.image_size
    );

    // Fill both method results for later output and comparison.
    UndistortionResult result;
    // Method one computes and samples the mapping internally.
    cv::undistort(
        image,
        result.direct,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        new_camera_matrix
    );

    // Method two precomputes floating-point maps suitable for repeated frames.
    cv::Mat map_x;
    cv::Mat map_y;
    cv::initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        cv::Mat(),
        new_camera_matrix,
        calibration.image_size,
        CV_32FC1,
        map_x,
        map_y
    );
    // Linear interpolation matches the direct workflow's intended sampling.
    cv::remap(
        image,
        result.remapped,
        map_x,
        map_y,
        cv::INTER_LINEAR
    );

    // Create nested output directories before either write.
    fs::create_directories(output_dir);
    // Fixed filenames make the method used for each image unambiguous.
    result.direct_path = output_dir / "undistorted_direct.jpg";
    result.remap_path = output_dir / "undistorted_remap.jpg";
    // imwrite reports some failures through false rather than an exception.
    if (!cv::imwrite(result.direct_path.string(), result.direct)) {
        throw std::runtime_error(
            "Could not write output image: " + result.direct_path.string()
        );
    }
    if (!cv::imwrite(result.remap_path.string(), result.remapped)) {
        throw std::runtime_error(
            "Could not write output image: " + result.remap_path.string()
        );
    }

    // Compare uncompressed matrices so JPEG encoding cannot affect the metric.
    cv::Mat difference;
    cv::absdiff(result.direct, result.remapped, difference);
    // imread produced BGR, so average the three per-channel means.
    const cv::Scalar channel_means = cv::mean(difference);
    result.mean_absolute_difference =
        (channel_means[0] + channel_means[1] + channel_means[2]) / 3.0;

    // Interactive mode exposes both results for visual inspection.
    if (display) {
        cv::imshow("Undistorted image (direct)", result.direct);
        cv::imshow("Undistorted image (remap)", result.remapped);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    // Return the complete images as well as their scalar difference.
    return result;
}

// Validate matrices, method agreement, and the readability of both saved files.
inline void validate_undistortion(
    const UndistortionResult& result,
    const double maximum_mean_difference
) {
    // Both workflows must return populated, same-sized color images.
    if (result.direct.empty() ||
        result.remapped.empty() ||
        result.direct.size() != result.remapped.size() ||
        result.direct.type() != result.remapped.type()) {
        throw std::runtime_error(
            "Undistortion methods returned empty or incompatible images"
        );
    }
    // A finite low mean difference confirms visually equivalent mappings.
    if (!std::isfinite(result.mean_absolute_difference) ||
        result.mean_absolute_difference >= maximum_mean_difference) {
        throw std::runtime_error(
            "Direct undistortion and remapping differ by more than expected"
        );
    }

    // Decode both written files so a successful encoder call alone is insufficient.
    const cv::Mat saved_direct = cv::imread(result.direct_path.string());
    const cv::Mat saved_remap = cv::imread(result.remap_path.string());
    // Each JPEG must remain readable at the exact generated matrix size.
    if (saved_direct.empty() ||
        saved_direct.size() != result.direct.size() ||
        saved_direct.type() != result.direct.type()) {
        throw std::runtime_error(
            "Saved direct-undistortion image is unreadable or has the wrong shape"
        );
    }
    if (saved_remap.empty() ||
        saved_remap.size() != result.remapped.size() ||
        saved_remap.type() != result.remapped.type()) {
        throw std::runtime_error(
            "Saved remap-undistortion image is unreadable or has the wrong shape"
        );
    }
}

}  // namespace tutorial
