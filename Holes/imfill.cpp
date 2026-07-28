#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct PipelineResult {
    cv::Mat binary;
    cv::Mat flooded_background;
    cv::Mat holes;
    cv::Mat filled;
};

cv::Mat thresholdForeground(const cv::Mat& gray, int threshold_value) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        throw std::invalid_argument(
            "thresholdForeground expects a non-empty 8-bit grayscale image");
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold must be in the range [0, 255]");
    }

    cv::Mat binary;
    cv::threshold(
        gray, binary, threshold_value, 255, cv::THRESH_BINARY_INV);
    return binary;
}

PipelineResult fillHoles(const cv::Mat& binary) {
    if (binary.empty() || binary.type() != CV_8UC1) {
        throw std::invalid_argument(
            "fillHoles expects a non-empty 8-bit single-channel mask");
    }

    cv::Mat invalid;
    cv::inRange(binary, cv::Scalar(1), cv::Scalar(254), invalid);
    if (cv::countNonZero(invalid) != 0) {
        throw std::invalid_argument(
            "fillHoles expects a binary mask containing only 0 and 255");
    }

    cv::Mat padded;
    cv::copyMakeBorder(
        binary, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat flooded_padded = padded.clone();
    cv::Mat flood_mask =
        cv::Mat::zeros(flooded_padded.rows + 2, flooded_padded.cols + 2, CV_8U);
    cv::floodFill(flooded_padded, flood_mask, cv::Point(0, 0), cv::Scalar(255));

    cv::Mat flooded_background =
        flooded_padded(cv::Rect(1, 1, binary.cols, binary.rows)).clone();
    cv::Mat holes;
    cv::bitwise_not(flooded_background, holes);
    cv::Mat filled;
    cv::bitwise_or(binary, holes, filled);
    return {binary.clone(), flooded_background, holes, filled};
}

void writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("OpenCV could not write " + path.string());
    }
}

void printUsage(const char* program) {
    std::cerr << "Usage: " << program
              << " [image] [--threshold 0..255] [--output-dir DIR] [--display]\n";
}

bool parseInteger(const std::string& value, int& parsed) {
    try {
        std::size_t consumed = 0;
        parsed = std::stoi(value, &consumed);
        return consumed == value.size();
    } catch (const std::exception&) {
        return false;
    }
}

int main(int argc, char** argv) {
    fs::path image_path = "nickel.jpg";
    fs::path output_directory = "output";
    int threshold_value = 220;
    bool display = false;
    bool positional_image_seen = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--threshold" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (!parseInteger(value, threshold_value)) {
                std::cerr << "error: invalid integer for --threshold: "
                          << value << '\n';
                return 2;
            }
        } else if (argument == "--output-dir" && index + 1 < argc) {
            output_directory = argv[++index];
        } else if (argument == "--display") {
            display = true;
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (!argument.empty() && argument.front() != '-' &&
                   !positional_image_seen) {
            image_path = argument;
            positional_image_seen = true;
        } else {
            printUsage(argv[0]);
            return 2;
        }
    }

    try {
        cv::Mat gray = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            throw std::runtime_error("Could not read input image: " +
                                     image_path.string());
        }

        const cv::Mat binary = thresholdForeground(gray, threshold_value);
        const PipelineResult result = fillHoles(binary);
        fs::create_directories(output_directory);
        writeImage(output_directory / "01-thresholded.png", result.binary);
        writeImage(
            output_directory / "02-flooded-background.png",
            result.flooded_background);
        writeImage(output_directory / "03-holes.png", result.holes);
        writeImage(output_directory / "04-filled.png", result.filled);

        std::cout << "input=" << image_path << '\n'
                  << "threshold=" << threshold_value << '\n'
                  << "foreground_pixels_before="
                  << cv::countNonZero(result.binary) << '\n'
                  << "filled_hole_pixels=" << cv::countNonZero(result.holes)
                  << '\n'
                  << "foreground_pixels_after="
                  << cv::countNonZero(result.filled) << '\n'
                  << "outputs=" << fs::absolute(output_directory) << '\n';

        if (display) {
            cv::imshow("Thresholded", result.binary);
            cv::imshow("Flooded exterior", result.flooded_background);
            cv::imshow("Holes", result.holes);
            cv::imshow("Filled", result.filled);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
