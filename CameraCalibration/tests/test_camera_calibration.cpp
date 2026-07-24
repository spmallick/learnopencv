// Exercise the same shared implementation used by both public executables.
#include "calibration_utils.hpp"

// Absolute differences compare results with the stable fixture oracle.
#include <cmath>
// Filesystem joins the directory argument with OpenCV's wildcard.
#include <filesystem>
// Test failures and the success summary are written to standard streams.
#include <iostream>
// Strings carry the final image-glob argument.
#include <string>

namespace {

// The repository intentionally tracks exactly 41 checkerboard images.
constexpr std::size_t kExpectedImages = 41;
// Every bundled image is 640 pixels wide.
constexpr int kExpectedWidth = 640;
// Every bundled image is 480 pixels high.
constexpr int kExpectedHeight = 480;
// Both OpenCV 4.14 and 5.0 produce this optimizer RMS.
constexpr double kExpectedRms = 0.2603201;
// The independently accumulated point RMSE rounds to this value.
constexpr double kExpectedReprojectionRmse = 0.26032;
// Small implementation-level differences are harmless below this error bound.
constexpr double kErrorTolerance = 0.01;
// Intrinsics remain stable to one percent and one hundredth of a pixel.
constexpr double kMatrixRelativeTolerance = 0.01;
constexpr double kMatrixAbsoluteTolerance = 0.01;

// Compare one camera-matrix element with combined relative/absolute tolerance.
bool nearly_equal(const double actual, const double expected) {
    // Scale the relative term by the expected value, then keep a floor near zero.
    const double tolerance = std::max(
        std::abs(expected) * kMatrixRelativeTolerance,
        kMatrixAbsoluteTolerance
    );
    // Values inside or on the tolerance boundary pass.
    return std::abs(actual - expected) <= tolerance;
}

}  // namespace

// Validate the complete bundled C++ numerical baseline.
int main(const int argc, char** argv) {
    // CTest passes the image directory, not a shell-dependent wildcard argument.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " image-directory\n";
        return 2;
    }

    try {
        // Construct the wildcard inside C++ so no shell expansion is involved.
        const std::string image_pattern =
            (std::filesystem::path(argv[1]) / "*.jpg").string();
        // Run the shared implementation without opening windows.
        const tutorial::CalibrationResult result =
            tutorial::calibrate_images(image_pattern, false);
        // Check generic shape, range, pose-count, and error invariants.
        tutorial::validate_calibration(result);

        // Missing files or failed detections must not pass with plausible values.
        if (result.image_paths.size() != kExpectedImages ||
            result.detected_images != kExpectedImages) {
            std::cerr
                << "Expected " << kExpectedImages
                << " readable checkerboards and detections, got "
                << result.image_paths.size() << " and "
                << result.detected_images << '\n';
            return 1;
        }
        // Size order is deliberately width by height in the OpenCV API.
        if (result.image_size != cv::Size(kExpectedWidth, kExpectedHeight)) {
            std::cerr
                << "Expected image size " << kExpectedWidth << 'x'
                << kExpectedHeight << ", got "
                << result.image_size << '\n';
            return 1;
        }
        // Validate OpenCV's optimizer error against the fixture oracle.
        if (std::abs(result.rms_error - kExpectedRms) >
            kErrorTolerance) {
            std::cerr
                << "Expected calibration RMS near " << kExpectedRms
                << ", got " << result.rms_error << '\n';
            return 1;
        }
        // Validate the independently recomputed reprojection error too.
        if (std::abs(
                result.reprojection_rmse -
                kExpectedReprojectionRmse
            ) > kErrorTolerance) {
            std::cerr
                << "Expected reprojection RMSE near "
                << kExpectedReprojectionRmse << ", got "
                << result.reprojection_rmse << '\n';
            return 1;
        }

        // This matrix was measured from all 41 tracked checkerboards.
        const cv::Matx33d expected_camera_matrix{
            503.5118, 0.0, 313.4135,
            0.0, 503.1461, 243.0911,
            0.0, 0.0, 1.0,
        };
        // Check every matrix position rather than only the focal lengths.
        for (int row = 0; row < expected_camera_matrix.rows; ++row) {
            for (int column = 0;
                 column < expected_camera_matrix.cols;
                 ++column) {
                const double actual =
                    result.camera_matrix.at<double>(row, column);
                const double expected =
                    expected_camera_matrix(row, column);
                if (!nearly_equal(actual, expected)) {
                    std::cerr
                        << "Camera matrix differs at (" << row << ", "
                        << column << "): expected " << expected
                        << ", got " << actual << '\n';
                    return 1;
                }
            }
        }

        // Print version and metrics as reviewable acceptance evidence.
        std::cout
            << "Camera calibration regression passed with OpenCV "
            << CV_VERSION << ": detections=" << result.detected_images
            << ", rms=" << result.rms_error
            << ", reprojection_rmse=" << result.reprojection_rmse
            << '\n';
    } catch (const std::exception& error) {
        // A thrown input, OpenCV, or validation error fails the regression.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    // Zero tells CTest that every fixture-specific comparison passed.
    return 0;
}
