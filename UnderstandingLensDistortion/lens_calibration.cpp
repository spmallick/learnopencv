#include "lens_calibration.hpp"

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib.hpp>
#include <opencv2/geometry/3d.hpp>
#include <opencv2/objdetect.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace lens_distortion {

namespace {

std::vector<cv::Point3f> objectTemplate(
    const cv::Size boardSize,
    const double squareSize) {
    if (boardSize.width < 2 || boardSize.height < 2) {
        throw std::invalid_argument(
            "Checkerboard dimensions must both be at least 2.");
    }
    if (!std::isfinite(squareSize) || squareSize <= 0.0) {
        throw std::invalid_argument(
            "squareSize must be finite and positive.");
    }

    std::vector<cv::Point3f> points;
    points.reserve(
        static_cast<std::size_t>(boardSize.width * boardSize.height));
    for (int row = 0; row < boardSize.height; ++row) {
        for (int column = 0; column < boardSize.width; ++column) {
            points.emplace_back(
                static_cast<float>(column * squareSize),
                static_cast<float>(row * squareSize),
                0.0F);
        }
    }
    return points;
}

bool hasJpegExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return extension == ".jpg" || extension == ".jpeg";
}

}  // namespace

std::vector<std::filesystem::path> discoverImages(
    const std::filesystem::path& directory) {
    if (!std::filesystem::exists(directory) ||
        !std::filesystem::is_directory(directory)) {
        throw std::runtime_error(
            "Calibration image directory does not exist: " +
            directory.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && hasJpegExtension(entry.path())) {
            paths.push_back(std::filesystem::absolute(entry.path()));
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error(
            "No JPEG calibration images found in: " + directory.string());
    }
    return paths;
}

CalibrationResult calibrateFromImages(
    const std::vector<std::filesystem::path>& imagePaths,
    const cv::Size boardSize,
    const double squareSize,
    const bool requireAll) {
    if (imagePaths.empty()) {
        throw std::invalid_argument(
            "At least one calibration image is required.");
    }

    const auto templatePoints = objectTemplate(boardSize, squareSize);
    std::vector<std::filesystem::path> sortedPaths = imagePaths;
    std::sort(sortedPaths.begin(), sortedPaths.end());

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    CalibrationResult result;
    const cv::TermCriteria criteria(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
    constexpr int flags =
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_FAST_CHECK |
        cv::CALIB_CB_NORMALIZE_IMAGE;

    for (const auto& path : sortedPaths) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error(
                "Could not read calibration image: " + path.string());
        }
        cv::Mat grayscale;
        cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
        const cv::Size currentSize(grayscale.cols, grayscale.rows);
        if (result.imageSize.empty()) {
            result.imageSize = currentSize;
        } else if (currentSize != result.imageSize) {
            throw std::invalid_argument(
                "Calibration image has size " +
                std::to_string(currentSize.width) + "x" +
                std::to_string(currentSize.height) + "; expected " +
                std::to_string(result.imageSize.width) + "x" +
                std::to_string(result.imageSize.height) + ": " +
                path.string());
        }

        std::vector<cv::Point2f> corners;
        const bool found = cv::findChessboardCorners(
            grayscale, boardSize, corners, flags);
        if (!found) {
            result.failedImages.push_back(path);
            continue;
        }

        cv::cornerSubPix(
            grayscale,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            criteria);
        objectPoints.push_back(templatePoints);
        imagePoints.push_back(corners);
        result.successfulImages.push_back(path);
        if (result.cornerPreview.empty()) {
            result.cornerPreview = image.clone();
            cv::drawChessboardCorners(
                result.cornerPreview, boardSize, corners, true);
        }
    }

    if (requireAll && !result.failedImages.empty()) {
        throw std::runtime_error(
            "Checkerboard detection failed for " +
            std::to_string(result.failedImages.size()) + " image(s).");
    }
    if (result.successfulImages.size() < 3U) {
        throw std::runtime_error(
            "At least three successful checkerboard views are required.");
    }

    result.rms = cv::calibrateCamera(
        objectPoints,
        imagePoints,
        result.imageSize,
        result.cameraMatrix,
        result.distortionCoefficients,
        result.rotationVectors,
        result.translationVectors);

    double squaredError = 0.0;
    std::size_t pointCount = 0U;
    for (std::size_t index = 0; index < objectPoints.size(); ++index) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(
            objectPoints[index],
            result.rotationVectors[index],
            result.translationVectors[index],
            result.cameraMatrix,
            result.distortionCoefficients,
            projected);
        squaredError += cv::norm(
            imagePoints[index], projected, cv::NORM_L2SQR);
        pointCount += projected.size();
    }
    result.reprojectionRMSE =
        std::sqrt(squaredError / static_cast<double>(pointCount));
    return result;
}

UndistortionResult undistortImage(
    const cv::Mat& image,
    const CalibrationResult& calibration,
    const double alpha,
    const std::string& method,
    const bool crop) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    if (image.size() != calibration.imageSize) {
        throw std::invalid_argument(
            "Input image size does not match the calibration size.");
    }
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("alpha must be finite and in [0, 1].");
    }

    UndistortionResult result;
    result.newCameraMatrix = cv::getOptimalNewCameraMatrix(
        calibration.cameraMatrix,
        calibration.distortionCoefficients,
        calibration.imageSize,
        alpha,
        calibration.imageSize,
        &result.roi);

    if (method == "direct") {
        cv::undistort(
            image,
            result.image,
            calibration.cameraMatrix,
            calibration.distortionCoefficients,
            result.newCameraMatrix);
    } else if (method == "remap") {
        cv::Mat mapX;
        cv::Mat mapY;
        cv::initUndistortRectifyMap(
            calibration.cameraMatrix,
            calibration.distortionCoefficients,
            cv::Mat(),
            result.newCameraMatrix,
            calibration.imageSize,
            CV_32FC1,
            mapX,
            mapY);
        cv::remap(
            image,
            result.image,
            mapX,
            mapY,
            cv::INTER_LINEAR);
    } else {
        throw std::invalid_argument(
            "method must be 'direct' or 'remap'.");
    }

    if (crop) {
        if (result.roi.width <= 0 || result.roi.height <= 0) {
            throw std::runtime_error(
                "OpenCV returned an empty undistortion ROI.");
        }
        result.image = result.image(result.roi).clone();
    }
    return result;
}

void saveCalibration(
    const CalibrationResult& calibration,
    const std::filesystem::path& destination) {
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    cv::FileStorage storage(destination.string(), cv::FileStorage::WRITE);
    if (!storage.isOpened()) {
        throw std::runtime_error(
            "Could not write calibration: " + destination.string());
    }
    storage << "imageWidth" << calibration.imageSize.width;
    storage << "imageHeight" << calibration.imageSize.height;
    storage << "rms" << calibration.rms;
    storage << "reprojectionRMSE" << calibration.reprojectionRMSE;
    storage << "cameraMatrix" << calibration.cameraMatrix;
    storage << "distortionCoefficients"
            << calibration.distortionCoefficients;
}

}  // namespace lens_distortion
