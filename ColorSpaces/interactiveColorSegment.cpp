#include "color_spaces.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef COLOR_SPACES_PROJECT_DIR
#error "COLOR_SPACES_PROJECT_DIR must be defined by CMake"
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
        std::filesystem::path(COLOR_SPACES_PROJECT_DIR) / "images" / "rub00.jpg";
    color_spaces::ColorSpace colorSpace = color_spaces::ColorSpace::Hsv;
    std::array<int, 3> lower{20, 80, 40};
    std::array<int, 3> upper{45, 255, 255};
    std::optional<std::filesystem::path> outputDirectory;
    bool noDisplay = false;
    bool validate = false;
    bool help = false;
};

std::array<int, 3> parseTriplet(const int argc, char** argv, int& index) {
    if (index + 3 >= argc) {
        throw std::invalid_argument("threshold option requires three integer values");
    }
    return {
        parseInteger(argv[++index]),
        parseInteger(argv[++index]),
        parseInteger(argv[++index]),
    };
}

Options parseArguments(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--input") {
            if (++index >= argc) {
                throw std::invalid_argument("--input requires a path");
            }
            options.input = std::filesystem::absolute(argv[index]);
        } else if (argument == "--space") {
            if (++index >= argc) {
                throw std::invalid_argument("--space requires BGR, HSV, YCrCb, or Lab");
            }
            options.colorSpace = color_spaces::parseColorSpace(argv[index]);
        } else if (argument == "--lower") {
            options.lower = parseTriplet(argc, argv, index);
        } else if (argument == "--upper") {
            options.upper = parseTriplet(argc, argv, index);
        } else if (argument == "--output-dir") {
            if (++index >= argc) {
                throw std::invalid_argument("--output-dir requires a path");
            }
            options.outputDirectory = std::filesystem::absolute(argv[index]);
        } else if (argument == "--no-display") {
            options.noDisplay = true;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return options;
}

struct Segmentation {
    cv::Mat mask;
    cv::Mat result;
};

Segmentation segment(const cv::Mat& image, const Options& options) {
    cv::Mat mask = color_spaces::thresholdMask(
        image, options.colorSpace, options.lower, options.upper);
    cv::Mat result = color_spaces::applyMask(image, mask);
    return Segmentation{mask, result};
}

std::pair<std::filesystem::path, std::filesystem::path> outputPaths(
    const Options& options) {
    if (!options.outputDirectory.has_value()) {
        throw std::logic_error("output directory is unavailable");
    }
    const std::string stem = options.input.stem().string();
    std::string normalizedSpace = color_spaces::name(options.colorSpace);
    for (char& character : normalizedSpace) {
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    }
    return {
        *options.outputDirectory / (stem + "-" + normalizedSpace + "-mask.png"),
        *options.outputDirectory / (stem + "-" + normalizedSpace + "-result.png"),
    };
}

void writeOutputs(const Options& options, const Segmentation& segmentation) {
    const auto [maskPath, resultPath] = outputPaths(options);
    color_spaces::writeImage(maskPath, segmentation.mask);
    color_spaces::writeImage(resultPath, segmentation.result);
    std::cout << "Wrote: " << maskPath << '\n';
    std::cout << "Wrote: " << resultPath << '\n';
}

void runValidation(const Options& options) {
    const cv::Mat image = color_spaces::readBgr(options.input);
    const Segmentation segmentation = segment(image, options);
    const int foregroundPixels = cv::countNonZero(segmentation.mask);
    if (foregroundPixels <= 0 ||
        foregroundPixels >= segmentation.mask.rows * segmentation.mask.cols) {
        throw std::runtime_error("foreground pixel count is outside the expected range");
    }
    if (segmentation.result.size() != image.size() ||
        segmentation.result.type() != image.type()) {
        throw std::runtime_error("segmentation result does not match input shape/type");
    }
    if (options.outputDirectory.has_value()) {
        writeOutputs(options, segmentation);
    }
    std::cout << "VALIDATION PASSED: foreground_pixels=" << foregroundPixels
              << ", image=" << image.cols << 'x' << image.rows << '\n';
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--input IMAGE] [--space BGR|HSV|YCrCb|Lab]"
        << " [--lower C0 C1 C2] [--upper C0 C1 C2]"
        << " [--output-dir DIR] [--no-display] [--validate]\n";
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

        const cv::Mat image = color_spaces::readBgr(options.input);
        const Segmentation segmentation = segment(image, options);
        const int foregroundPixels = cv::countNonZero(segmentation.mask);
        std::cout << "Input: " << options.input << '\n';
        std::cout << "Color space: " << color_spaces::name(options.colorSpace) << '\n';
        std::cout << "Foreground pixels: " << foregroundPixels << '/'
                  << segmentation.mask.rows * segmentation.mask.cols << '\n';

        if (options.outputDirectory.has_value()) {
            writeOutputs(options, segmentation);
        }
        if (!options.noDisplay) {
            cv::imshow("Input", image);
            cv::imshow("Mask", segmentation.mask);
            cv::imshow(
                color_spaces::name(options.colorSpace) + " segmentation",
                segmentation.result);
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
