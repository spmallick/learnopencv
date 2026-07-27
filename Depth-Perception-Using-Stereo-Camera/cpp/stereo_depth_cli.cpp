#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "stereo_depth.hpp"

namespace {

struct Options {
    std::filesystem::path left;
    std::filesystem::path right;
    std::filesystem::path maps =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "stereo_rectify_maps.xml";
    std::filesystem::path config =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "depth_estimation_params_cpp.xml";
    std::filesystem::path outputDirectory =
        std::filesystem::path(STEREO_SOURCE_DIR) / "outputs";
    bool alreadyRectified = false;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto requireValue = [&](const char* option) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    std::string("Missing value for ") + option);
            }
            return argv[++index];
        };
        if (argument == "--left") {
            options.left = requireValue("--left");
        } else if (argument == "--right") {
            options.right = requireValue("--right");
        } else if (argument == "--maps") {
            options.maps = requireValue("--maps");
        } else if (argument == "--config") {
            options.config = requireValue("--config");
        } else if (argument == "--output-dir") {
            options.outputDirectory =
                requireValue("--output-dir");
        } else if (argument == "--already-rectified") {
            options.alreadyRectified = true;
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }
    if (options.left.empty() || options.right.empty()) {
        throw std::invalid_argument(
            "--left and --right image paths are required.");
    }
    return options;
}

cv::Mat depthVisualization(const cv::Mat& depth) {
    cv::Mat validMask = depth == depth;
    cv::Mat gray = cv::Mat::zeros(depth.size(), CV_8UC1);
    if (cv::countNonZero(validMask) > 0) {
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(
            depth, &minimum, &maximum, nullptr, nullptr, validMask);
        if (maximum > minimum) {
            for (int row = 0; row < depth.rows; ++row) {
                const auto* values = depth.ptr<float>(row);
                auto* pixels = gray.ptr<std::uint8_t>(row);
                for (int col = 0; col < depth.cols; ++col) {
                    if (std::isfinite(values[col])) {
                        pixels[col] = cv::saturate_cast<std::uint8_t>(
                            (maximum - values[col]) * 255.0 /
                            (maximum - minimum));
                    }
                }
            }
        }
    }
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    return color;
}

void writeImage(
    const std::filesystem::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error(
            "Could not write output image: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        cv::Mat left =
            cv::imread(options.left.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat right =
            cv::imread(options.right.string(), cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty()) {
            throw std::runtime_error(
                "Could not read both grayscale stereo images.");
        }

        const auto config =
            learnopencv::stereo::loadConfig(options.config);
        if (!options.alreadyRectified) {
            const auto maps =
                learnopencv::stereo::loadRectificationMaps(options.maps);
            std::tie(left, right) =
                learnopencv::stereo::rectifyPair(left, right, maps);
        }
        const cv::Mat disparity =
            learnopencv::stereo::computeDisparity(
                left, right, config);
        const cv::Mat depth =
            learnopencv::stereo::disparityToDepth(
                disparity, config);

        std::filesystem::create_directories(
            options.outputDirectory);
        writeImage(
            options.outputDirectory / "left-rectified.png", left);
        writeImage(
            options.outputDirectory / "right-rectified.png", right);
        writeImage(
            options.outputDirectory / "disparity.png",
            learnopencv::stereo::disparityVisualization(disparity));
        writeImage(
            options.outputDirectory / "depth.png",
            depthVisualization(depth));

        const cv::Mat validMask = disparity == disparity;
        const double validFraction =
            static_cast<double>(cv::countNonZero(validMask)) /
            static_cast<double>(disparity.total());
        std::cout << "Valid disparity fraction: " << validFraction
                  << '\n'
                  << "Saved outputs under: "
                  << options.outputDirectory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
