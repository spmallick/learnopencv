#include "hough_common.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef HOUGH_PROJECT_DIR
#error "HOUGH_PROJECT_DIR must be defined by CMake"
#endif

namespace {

double parseDouble(const std::string& text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite number: " + text);
    }
    return value;
}

int parseInteger(const std::string& text) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid integer: " + text);
    }
    return value;
}

struct Options {
    std::filesystem::path input =
        std::filesystem::path(HOUGH_PROJECT_DIR) / "brown-eyes.jpg";
    std::optional<std::filesystem::path> outputDirectory;
    double dp = 1.2;
    std::optional<double> minimumDistance;
    double param1 = 120.0;
    double param2 = 30.0;
    int minimumRadius = 25;
    int maximumRadius = 55;
    bool noDisplay = false;
    bool validate = false;
    bool help = false;
};

Options parseArguments(const int argc, char** argv) {
    Options options;
    bool inputWasSet = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--output-dir") {
            if (++index >= argc) {
                throw std::invalid_argument("--output-dir requires a path");
            }
            options.outputDirectory = std::filesystem::absolute(argv[index]);
        } else if (argument == "--dp") {
            if (++index >= argc) {
                throw std::invalid_argument("--dp requires a number");
            }
            options.dp = parseDouble(argv[index]);
        } else if (argument == "--min-distance") {
            if (++index >= argc) {
                throw std::invalid_argument("--min-distance requires a number");
            }
            options.minimumDistance = parseDouble(argv[index]);
        } else if (argument == "--param1") {
            if (++index >= argc) {
                throw std::invalid_argument("--param1 requires a number");
            }
            options.param1 = parseDouble(argv[index]);
        } else if (argument == "--param2") {
            if (++index >= argc) {
                throw std::invalid_argument("--param2 requires a number");
            }
            options.param2 = parseDouble(argv[index]);
        } else if (argument == "--min-radius") {
            if (++index >= argc) {
                throw std::invalid_argument("--min-radius requires an integer");
            }
            options.minimumRadius = parseInteger(argv[index]);
        } else if (argument == "--max-radius") {
            if (++index >= argc) {
                throw std::invalid_argument("--max-radius requires an integer");
            }
            options.maximumRadius = parseInteger(argv[index]);
        } else if (argument == "--no-display") {
            options.noDisplay = true;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (!argument.empty() && argument[0] == '-') {
            throw std::invalid_argument("unknown argument: " + argument);
        } else if (!inputWasSet) {
            options.input = std::filesystem::absolute(argument);
            inputWasSet = true;
        } else {
            throw std::invalid_argument("only one input image may be provided");
        }
    }
    return options;
}

std::pair<std::filesystem::path, std::filesystem::path> outputPaths(
    const Options& options) {
    if (!options.outputDirectory.has_value()) {
        throw std::logic_error("output directory is unavailable");
    }
    const std::string stem = options.input.stem().string();
    return {
        *options.outputDirectory / (stem + "-blurred.png"),
        *options.outputDirectory / (stem + "-circles.png"),
    };
}

void writeOutputs(
    const Options& options,
    const hough::CircleDetection& detection,
    const cv::Mat& annotated) {
    const auto [blurPath, resultPath] = outputPaths(options);
    hough::writeImage(blurPath, detection.blurred);
    hough::writeImage(resultPath, annotated);
    std::cout << "Wrote: " << blurPath << '\n';
    std::cout << "Wrote: " << resultPath << '\n';
}

void runValidation(const Options& options) {
    cv::Mat synthetic = cv::Mat::zeros(256, 256, CV_8UC3);
    cv::circle(
        synthetic,
        cv::Point(128, 128),
        60,
        cv::Scalar(255, 255, 255),
        3,
        cv::LINE_AA);
    const hough::CircleDetection syntheticDetection =
        hough::detectCircles(synthetic, 1.2, 40.0, 120.0, 30.0, 50, 70);
    if (syntheticDetection.circles.empty()) {
        throw std::runtime_error("synthetic circle was not detected");
    }

    bool accurateCircleFound = false;
    for (const cv::Vec3f& circleValue : syntheticDetection.circles) {
        const double centerError =
            std::hypot(circleValue[0] - 128.0F, circleValue[1] - 128.0F);
        const double radiusError = std::abs(circleValue[2] - 60.0F);
        if (centerError <= 4.0 && radiusError <= 4.0) {
            accurateCircleFound = true;
        }
    }
    if (!accurateCircleFound) {
        throw std::runtime_error("synthetic circle geometry was inaccurate");
    }

    const cv::Mat image = hough::readBgr(options.input);
    const hough::CircleDetection detection =
        hough::detectCircles(image, 1.2, image.rows / 4.0, 120.0, 30.0, 25, 55);
    if (detection.circles.empty()) {
        throw std::runtime_error("no circles were detected in brown-eyes.jpg");
    }
    const cv::Mat annotated = hough::drawCircles(image, detection.circles);
    if (options.outputDirectory.has_value()) {
        writeOutputs(options, detection, annotated);
    }

    std::cout << "VALIDATION PASSED: circles=" << detection.circles.size()
              << ", synthetic_circles=" << syntheticDetection.circles.size() << '\n';
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [IMAGE] [--output-dir DIR] [--dp N] [--min-distance N]"
        << " [--param1 N] [--param2 N] [--min-radius N] [--max-radius N]"
        << " [--no-display] [--validate]\n";
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        if (options.help) {
            printUsage(argv[0]);
            return 0;
        }
        if (options.validate) {
            runValidation(options);
            return 0;
        }

        const cv::Mat image = hough::readBgr(options.input);
        const double minimumDistance =
            options.minimumDistance.value_or(image.rows / 4.0);
        const hough::CircleDetection detection = hough::detectCircles(
            image,
            options.dp,
            minimumDistance,
            options.param1,
            options.param2,
            options.minimumRadius,
            options.maximumRadius);
        const cv::Mat annotated = hough::drawCircles(image, detection.circles);
        std::cout << "Input: " << options.input << '\n';
        std::cout << "Circles: " << detection.circles.size() << '\n';

        if (options.outputDirectory.has_value()) {
            writeOutputs(options, detection, annotated);
        }
        if (!options.noDisplay) {
            cv::imshow("Median-blurred grayscale", detection.blurred);
            cv::imshow("Hough circles", annotated);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        printUsage(argv[0]);
        return 2;
    }
}
