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
        std::filesystem::path(HOUGH_PROJECT_DIR) / "lanes.jpg";
    std::optional<std::filesystem::path> outputDirectory;
    int cannyLow = 50;
    int cannyHigh = 150;
    int houghThreshold = 50;
    int minimumLineLength = 40;
    int maximumLineGap = 25;
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
        } else if (argument == "--canny-low") {
            if (++index >= argc) {
                throw std::invalid_argument("--canny-low requires an integer");
            }
            options.cannyLow = parseInteger(argv[index]);
        } else if (argument == "--canny-high") {
            if (++index >= argc) {
                throw std::invalid_argument("--canny-high requires an integer");
            }
            options.cannyHigh = parseInteger(argv[index]);
        } else if (argument == "--hough-threshold") {
            if (++index >= argc) {
                throw std::invalid_argument("--hough-threshold requires an integer");
            }
            options.houghThreshold = parseInteger(argv[index]);
        } else if (argument == "--min-line-length") {
            if (++index >= argc) {
                throw std::invalid_argument("--min-line-length requires an integer");
            }
            options.minimumLineLength = parseInteger(argv[index]);
        } else if (argument == "--max-line-gap") {
            if (++index >= argc) {
                throw std::invalid_argument("--max-line-gap requires an integer");
            }
            options.maximumLineGap = parseInteger(argv[index]);
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

bool hasHorizontalLine(
    const std::vector<cv::Vec4i>& lines,
    const int minimumLength = 150) {
    for (const cv::Vec4i& line : lines) {
        if (std::abs(line[3] - line[1]) <= 3 &&
            std::abs(line[2] - line[0]) >= minimumLength) {
            return true;
        }
    }
    return false;
}

bool hasDiagonalLine(
    const std::vector<cv::Vec4i>& lines,
    const double minimumLength = 130.0) {
    for (const cv::Vec4i& line : lines) {
        const int deltaX = line[2] - line[0];
        const int deltaY = line[3] - line[1];
        if (std::hypot(deltaX, deltaY) >= minimumLength &&
            std::abs(deltaX) > 20 &&
            std::abs(deltaY) > 20) {
            return true;
        }
    }
    return false;
}

std::pair<std::filesystem::path, std::filesystem::path> outputPaths(
    const Options& options) {
    if (!options.outputDirectory.has_value()) {
        throw std::logic_error("output directory is unavailable");
    }
    const std::string stem = options.input.stem().string();
    return {
        *options.outputDirectory / (stem + "-edges.png"),
        *options.outputDirectory / (stem + "-lines.png"),
    };
}

void writeOutputs(
    const Options& options,
    const hough::LineDetection& detection,
    const cv::Mat& annotated) {
    const auto [edgePath, resultPath] = outputPaths(options);
    hough::writeImage(edgePath, detection.edges);
    hough::writeImage(resultPath, annotated);
    std::cout << "Wrote: " << edgePath << '\n';
    std::cout << "Wrote: " << resultPath << '\n';
}

void runValidation(const Options& options) {
    cv::Mat synthetic = cv::Mat::zeros(256, 256, CV_8UC3);
    cv::line(
        synthetic,
        cv::Point(20, 45),
        cv::Point(235, 45),
        cv::Scalar(255, 255, 255),
        3,
        cv::LINE_AA);
    cv::line(
        synthetic,
        cv::Point(25, 225),
        cv::Point(225, 75),
        cv::Scalar(255, 255, 255),
        3,
        cv::LINE_AA);
    const hough::LineDetection syntheticDetection =
        hough::detectLines(synthetic, 50, 150, 45, 100, 12);
    if (!hasHorizontalLine(syntheticDetection.lines) ||
        !hasDiagonalLine(syntheticDetection.lines)) {
        throw std::runtime_error("synthetic semantic lines were not recovered");
    }

    const cv::Mat image = hough::readBgr(options.input);
    const hough::LineDetection detection = hough::detectLines(image);
    if (detection.lines.empty()) {
        throw std::runtime_error("no lines were detected in lanes.jpg");
    }
    const cv::Mat annotated = hough::drawLines(image, detection.lines);
    if (options.outputDirectory.has_value()) {
        writeOutputs(options, detection, annotated);
    }

    std::cout << "VALIDATION PASSED: lines=" << detection.lines.size()
              << ", edge_pixels=" << cv::countNonZero(detection.edges)
              << ", synthetic_lines=" << syntheticDetection.lines.size() << '\n';
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [IMAGE] [--output-dir DIR] [--canny-low N] [--canny-high N]"
        << " [--hough-threshold N] [--min-line-length N] [--max-line-gap N]"
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
        const hough::LineDetection detection = hough::detectLines(
            image,
            options.cannyLow,
            options.cannyHigh,
            options.houghThreshold,
            options.minimumLineLength,
            options.maximumLineGap);
        const cv::Mat annotated = hough::drawLines(image, detection.lines);
        std::cout << "Input: " << options.input << '\n';
        std::cout << "Line segments: " << detection.lines.size() << '\n';
        std::cout << "Edge pixels: " << cv::countNonZero(detection.edges) << '\n';

        if (options.outputDirectory.has_value()) {
            writeOutputs(options, detection, annotated);
        }
        if (!options.noDisplay) {
            cv::imshow("Edges", detection.edges);
            cv::imshow("Probabilistic Hough lines", annotated);
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
