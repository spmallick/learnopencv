#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "stereo_depth.hpp"

namespace {

constexpr const char* kWindow = "StereoBM disparity";

struct Options {
    int leftCamera = 2;
    int rightCamera = 0;
    int maxFrames = 0;
    bool display = true;
    std::filesystem::path maps =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "stereo_rectify_maps.xml";
    std::filesystem::path config =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "depth_estimation_params_cpp.xml";
    std::filesystem::path saveConfig;
    std::filesystem::path output;
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
        if (argument == "--left-camera") {
            options.leftCamera =
                std::stoi(requireValue("--left-camera"));
        } else if (argument == "--right-camera") {
            options.rightCamera =
                std::stoi(requireValue("--right-camera"));
        } else if (argument == "--maps") {
            options.maps = requireValue("--maps");
        } else if (argument == "--config") {
            options.config = requireValue("--config");
        } else if (argument == "--save-config") {
            options.saveConfig = requireValue("--save-config");
        } else if (argument == "--output") {
            options.output = requireValue("--output");
        } else if (argument == "--max-frames") {
            options.maxFrames =
                std::stoi(requireValue("--max-frames"));
        } else if (argument == "--no-display") {
            options.display = false;
        } else {
            throw std::invalid_argument(
                "Unknown option: " + argument);
        }
    }
    if (options.leftCamera == options.rightCamera) {
        throw std::invalid_argument(
            "Left and right camera indices must be different.");
    }
    if (options.maxFrames < 0) {
        throw std::invalid_argument(
            "--max-frames must be non-negative.");
    }
    if (!options.display && options.maxFrames == 0) {
        throw std::invalid_argument(
            "--no-display requires a positive --max-frames.");
    }
    return options;
}

void noOp(int, void*) {}

void createControls(
    const learnopencv::stereo::StereoBMConfig& config) {
    cv::namedWindow(kWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(kWindow, 800, 600);
    const auto add = [](const char* name, int value, int maximum) {
        cv::createTrackbar(name, kWindow, nullptr, maximum, noOp);
        cv::setTrackbarPos(name, kWindow, value);
    };
    add("numDisparities/16", config.numDisparities / 16, 32);
    add("blockSize", (config.blockSize - 5) / 2, 125);
    add("preFilterType", config.preFilterType, 1);
    add("preFilterSize", (config.preFilterSize - 5) / 2, 125);
    add("preFilterCap", config.preFilterCap, 63);
    add("textureThreshold", config.textureThreshold, 100);
    add("uniquenessRatio", config.uniquenessRatio, 100);
    add("speckleRange", config.speckleRange, 100);
    add("speckleWindowSize", config.speckleWindowSize, 200);
    add("disp12MaxDiff+1", config.disp12MaxDiff + 1, 65);
    add("minDisparity", std::max(0, config.minDisparity), 64);
}

learnopencv::stereo::StereoBMConfig configFromControls(
    learnopencv::stereo::StereoBMConfig config) {
    config.numDisparities =
        std::max(1, cv::getTrackbarPos("numDisparities/16", kWindow)) *
        16;
    config.blockSize =
        cv::getTrackbarPos("blockSize", kWindow) * 2 + 5;
    config.preFilterType =
        cv::getTrackbarPos("preFilterType", kWindow);
    config.preFilterSize =
        cv::getTrackbarPos("preFilterSize", kWindow) * 2 + 5;
    config.preFilterCap =
        std::max(1, cv::getTrackbarPos("preFilterCap", kWindow));
    config.textureThreshold =
        cv::getTrackbarPos("textureThreshold", kWindow);
    config.uniquenessRatio =
        cv::getTrackbarPos("uniquenessRatio", kWindow);
    config.speckleRange =
        cv::getTrackbarPos("speckleRange", kWindow);
    config.speckleWindowSize =
        cv::getTrackbarPos("speckleWindowSize", kWindow);
    config.disp12MaxDiff =
        cv::getTrackbarPos("disp12MaxDiff+1", kWindow) - 1;
    config.minDisparity =
        cv::getTrackbarPos("minDisparity", kWindow);
    config.validate();
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        auto config = learnopencv::stereo::loadConfig(options.config);
        const auto maps =
            learnopencv::stereo::loadRectificationMaps(options.maps);
        auto matcher = learnopencv::stereo::createMatcher(config);

        cv::VideoCapture leftCamera(options.leftCamera);
        cv::VideoCapture rightCamera(options.rightCamera);
        if (!leftCamera.isOpened()) {
            throw std::runtime_error(
                "Could not open left camera " +
                std::to_string(options.leftCamera));
        }
        if (!rightCamera.isOpened()) {
            throw std::runtime_error(
                "Could not open right camera " +
                std::to_string(options.rightCamera));
        }
        if (options.display) {
            createControls(config);
        }

        cv::Mat finalView;
        int frameCount = 0;
        while (true) {
            if (!leftCamera.grab() || !rightCamera.grab()) {
                throw std::runtime_error(
                    "Could not grab synchronized stereo frames.");
            }
            cv::Mat leftColor;
            cv::Mat rightColor;
            if (!leftCamera.retrieve(leftColor) ||
                !rightCamera.retrieve(rightColor) ||
                leftColor.empty() || rightColor.empty()) {
                throw std::runtime_error(
                    "Could not retrieve synchronized stereo frames.");
            }

            if (options.display) {
                const auto updated = configFromControls(config);
                if (
                    updated.numDisparities != config.numDisparities ||
                    updated.blockSize != config.blockSize ||
                    updated.preFilterType != config.preFilterType ||
                    updated.preFilterSize != config.preFilterSize ||
                    updated.preFilterCap != config.preFilterCap ||
                    updated.textureThreshold != config.textureThreshold ||
                    updated.uniquenessRatio != config.uniquenessRatio ||
                    updated.speckleRange != config.speckleRange ||
                    updated.speckleWindowSize != config.speckleWindowSize ||
                    updated.disp12MaxDiff != config.disp12MaxDiff ||
                    updated.minDisparity != config.minDisparity) {
                    config = updated;
                    matcher =
                        learnopencv::stereo::createMatcher(config);
                }
            }

            cv::Mat leftGray;
            cv::Mat rightGray;
            cv::cvtColor(
                leftColor, leftGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(
                rightColor, rightGray, cv::COLOR_BGR2GRAY);
            const auto [leftRectified, rightRectified] =
                learnopencv::stereo::rectifyPair(
                    leftGray, rightGray, maps);
            const cv::Mat disparity =
                learnopencv::stereo::computeDisparity(
                    leftRectified, rightRectified, config, matcher);
            finalView =
                learnopencv::stereo::disparityVisualization(disparity);
            ++frameCount;

            if (options.display) {
                cv::imshow(kWindow, finalView);
                if (cv::waitKey(1) == 27) {
                    break;
                }
            }
            if (
                options.maxFrames > 0 &&
                frameCount >= options.maxFrames) {
                break;
            }
        }
        if (options.display) {
            cv::destroyAllWindows();
        }
        if (finalView.empty()) {
            throw std::runtime_error("No stereo frames were processed.");
        }
        if (!options.output.empty()) {
            if (!options.output.parent_path().empty()) {
                std::filesystem::create_directories(
                    options.output.parent_path());
            }
            if (!cv::imwrite(options.output.string(), finalView)) {
                throw std::runtime_error(
                    "Could not write disparity output.");
            }
            std::cout << "Saved final disparity view: "
                      << options.output << '\n';
        }
        if (!options.saveConfig.empty()) {
            learnopencv::stereo::saveConfig(
                config, options.saveConfig);
            std::cout << "Saved configuration: "
                      << options.saveConfig << '\n';
        }
        std::cout << "Processed stereo frames: " << frameCount << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
