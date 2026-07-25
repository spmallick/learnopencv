#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path input = "images/emir.jpg";
    fs::path output_dir = "output";
    std::string motion = "homography";
    int iterations = 5000;
    double epsilon = 1e-7;
    bool display = true;
    bool validate = false;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--input" && i + 1 < argc) {
            options.input = argv[++i];
        } else if (argument == "--output-dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (argument == "--motion" && i + 1 < argc) {
            options.motion = argv[++i];
        } else if (argument == "--iterations" && i + 1 < argc) {
            options.iterations = std::stoi(argv[++i]);
        } else if (argument == "--epsilon" && i + 1 < argc) {
            options.epsilon = std::stod(argv[++i]);
        } else if (argument == "--no-display") {
            options.display = false;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--help") {
            std::cout
                << "Usage: image_alignment [--input FILE] "
                   "[--motion translation|euclidean|affine|homography] "
                   "[--output-dir DIR] [--no-display] [--validate]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

cv::Mat gradient(const cv::Mat& image) {
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(image, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::addWeighted(cv::abs(grad_x), 0.5, cv::abs(grad_y), 0.5, 0.0,
                    magnitude);
    return magnitude;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const std::map<std::string, int> motion_models = {
            {"translation", cv::MOTION_TRANSLATION},
            {"euclidean", cv::MOTION_EUCLIDEAN},
            {"affine", cv::MOTION_AFFINE},
            {"homography", cv::MOTION_HOMOGRAPHY},
        };
        const auto motion_iterator = motion_models.find(options.motion);
        if (motion_iterator == motion_models.end()) {
            throw std::invalid_argument("Unknown motion model: " + options.motion);
        }
        const int motion_model = motion_iterator->second;

        cv::Mat stacked = cv::imread(options.input.string(), cv::IMREAD_GRAYSCALE);
        if (stacked.empty()) {
            throw std::runtime_error("Could not read input image: " +
                                     options.input.string());
        }
        const int height = stacked.rows / 3;
        if (height < 1) {
            throw std::runtime_error("Input does not contain three stacked channels");
        }
        const int width = stacked.cols;
        stacked = stacked(cv::Rect(0, 0, width, height * 3));
        std::vector<cv::Mat> channels = {
            stacked(cv::Rect(0, 0, width, height)).clone(),
            stacked(cv::Rect(0, height, width, height)).clone(),
            stacked(cv::Rect(0, 2 * height, width, height)).clone(),
        };
        cv::Mat unaligned;
        cv::merge(channels, unaligned);

        std::vector<cv::Mat> aligned_channels(3);
        std::vector<double> correlations;
        const cv::TermCriteria criteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS, options.iterations,
            options.epsilon);
        const cv::Mat template_gradient = gradient(channels[2]);
        for (int index = 0; index < 2; ++index) {
            cv::Mat warp = motion_model == cv::MOTION_HOMOGRAPHY
                               ? cv::Mat::eye(3, 3, CV_32F)
                               : cv::Mat::eye(2, 3, CV_32F);
            const double correlation = cv::findTransformECC(
                template_gradient, gradient(channels[index]), warp, motion_model,
                criteria);
            correlations.push_back(correlation);
            const int flags = cv::INTER_LINEAR | cv::WARP_INVERSE_MAP;
            if (motion_model == cv::MOTION_HOMOGRAPHY) {
                cv::warpPerspective(channels[index], aligned_channels[index],
                                    warp, channels[2].size(), flags);
            } else {
                cv::warpAffine(channels[index], aligned_channels[index], warp,
                               channels[2].size(), flags);
            }
            std::cout << (index == 0 ? "Blue" : "Green")
                      << " ECC correlation: " << correlation << "\n"
                      << "Warp matrix:\n" << warp << "\n";
            if (!cv::checkRange(warp)) {
                throw std::runtime_error("ECC returned a non-finite warp");
            }
        }
        aligned_channels[2] = channels[2].clone();
        cv::Mat aligned;
        cv::merge(aligned_channels, aligned);

        fs::create_directories(options.output_dir);
        const fs::path before_path =
            options.output_dir / "stacked-color-unaligned.jpg";
        const fs::path after_path =
            options.output_dir / "stacked-color-aligned.jpg";
        if (!cv::imwrite(before_path.string(), unaligned) ||
            !cv::imwrite(after_path.string(), aligned)) {
            throw std::runtime_error("Could not write color output images");
        }

        std::cout << "OpenCV: " << CV_VERSION << "\n"
                  << "Motion model: " << options.motion << "\n"
                  << "Saved: " << after_path << "\n";
        if (options.validate) {
            if (!std::isfinite(correlations[0]) ||
                !std::isfinite(correlations[1]) ||
                std::min(correlations[0], correlations[1]) <= 0.0) {
                throw std::runtime_error("Stacked-channel ECC validation failed");
            }
            std::cout << "Validation: PASS\n";
        }
        if (options.display) {
            cv::imshow("Unaligned color", unaligned);
            cv::imshow("ECC-aligned color", aligned);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
