#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path input = "headPose.jpg";
    fs::path output_dir = "output";
    bool display = true;
    bool validate = false;
    double focal_length = 0.0;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--input" && i + 1 < argc) {
            options.input = argv[++i];
        } else if (argument == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (argument == "--focal-length" && i + 1 < argc) {
            options.focal_length = std::stod(argv[++i]);
        } else if (argument == "--no-display") {
            options.display = false;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--help") {
            std::cout
                << "Usage: head_pose [--input IMAGE] [--output-dir DIR] "
                   "[--focal-length PIXELS] [--no-display] [--validate]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        cv::Mat image = cv::imread(options.input.string());
        if (image.empty()) {
            throw std::runtime_error("Could not read input image: " + options.input.string());
        }

        const std::vector<cv::Point2d> image_points = {
            {359.0, 391.0}, {399.0, 561.0}, {337.0, 297.0},
            {513.0, 301.0}, {345.0, 465.0}, {453.0, 469.0},
        };
        const std::vector<cv::Point3d> model_points = {
            {0.0, 0.0, 0.0},       {0.0, -330.0, -65.0},
            {-225.0, 170.0, -135.0}, {225.0, 170.0, -135.0},
            {-150.0, -150.0, -125.0}, {150.0, -150.0, -125.0},
        };

        const double focal =
            options.focal_length > 0.0 ? options.focal_length : image.cols;
        const cv::Point2d center(image.cols / 2.0, image.rows / 2.0);
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = focal;
        camera_matrix.at<double>(1, 1) = focal;
        camera_matrix.at<double>(0, 2) = center.x;
        camera_matrix.at<double>(1, 2) = center.y;
        const cv::Mat distortion = cv::Mat::zeros(4, 1, CV_64F);
        cv::Mat rotation_vector;
        cv::Mat translation_vector;
        const bool success = cv::solvePnP(
            model_points, image_points, camera_matrix, distortion,
            rotation_vector, translation_vector, false, cv::SOLVEPNP_ITERATIVE);
        if (!success) {
            throw std::runtime_error("solvePnP could not estimate a pose");
        }

        std::vector<cv::Point2d> reprojected;
        cv::projectPoints(model_points, rotation_vector, translation_vector,
                          camera_matrix, distortion, reprojected);
        double squared_error = 0.0;
        for (std::size_t i = 0; i < image_points.size(); ++i) {
            const cv::Point2d residual = reprojected[i] - image_points[i];
            squared_error += residual.dot(residual);
        }
        const double rmse = std::sqrt(squared_error / image_points.size());

        std::vector<cv::Point2d> nose_end;
        cv::projectPoints(std::vector<cv::Point3d>{{0.0, 0.0, 1000.0}},
                          rotation_vector, translation_vector, camera_matrix,
                          distortion, nose_end);
        cv::Mat result = image.clone();
        for (const auto& point : image_points) {
            cv::circle(result, point, 4, cv::Scalar(0, 0, 255), -1,
                       cv::LINE_AA);
        }
        cv::line(result, image_points.front(), nose_end.front(),
                 cv::Scalar(255, 0, 0), 3, cv::LINE_AA);

        fs::create_directories(options.output_dir);
        const fs::path output_path = options.output_dir / "head-pose-result.jpg";
        if (!cv::imwrite(output_path.string(), result)) {
            throw std::runtime_error("Could not write output image");
        }

        std::cout << "OpenCV: " << CV_VERSION << "\n"
                  << "Camera matrix:\n" << camera_matrix << "\n"
                  << "Rotation vector:\n" << rotation_vector << "\n"
                  << "Translation vector:\n" << translation_vector << "\n"
                  << "Reprojection RMSE: " << std::fixed << std::setprecision(3)
                  << rmse << " pixels\n"
                  << "Saved: " << output_path << "\n";

        if (options.validate) {
            if (!cv::checkRange(rotation_vector) ||
                !cv::checkRange(translation_vector) || !std::isfinite(rmse) ||
                rmse > 50.0) {
                throw std::runtime_error("Head-pose validation failed");
            }
            const cv::Mat check = cv::imread(output_path.string());
            if (check.empty() || check.size() != image.size()) {
                throw std::runtime_error("Saved output validation failed");
            }
            std::cout << "Validation: PASS\n";
        }

        if (options.display) {
            cv::imshow("Head pose", result);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
