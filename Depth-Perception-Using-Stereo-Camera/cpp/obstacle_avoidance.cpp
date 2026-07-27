#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "stereo_depth.hpp"

namespace {

struct Options {
    int leftCamera = 2;
    int rightCamera = 0;
    int maxFrames = 0;
    bool display = true;
    double minDepth = 10.0;
    double safeDistance = 100.0;
    double minimumAreaFraction = 0.01;
    std::filesystem::path maps =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "stereo_rectify_maps.xml";
    std::filesystem::path config =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "depth_estimation_params_cpp.xml";
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
        } else if (argument == "--minimum-depth") {
            options.minDepth =
                std::stod(requireValue("--minimum-depth"));
        } else if (argument == "--safe-distance") {
            options.safeDistance =
                std::stod(requireValue("--safe-distance"));
        } else if (argument == "--minimum-area-fraction") {
            options.minimumAreaFraction =
                std::stod(requireValue("--minimum-area-fraction"));
        } else if (argument == "--max-frames") {
            options.maxFrames =
                std::stoi(requireValue("--max-frames"));
        } else if (argument == "--output") {
            options.output = requireValue("--output");
        } else if (argument == "--no-display") {
            options.display = false;
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const auto config =
            learnopencv::stereo::loadConfig(options.config);
        const auto maps =
            learnopencv::stereo::loadRectificationMaps(options.maps);
        const auto matcher =
            learnopencv::stereo::createMatcher(config);

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

        cv::Mat finalCanvas;
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
            const cv::Mat depth =
                learnopencv::stereo::disparityToDepth(
                    disparity, config);
            const auto [mask, obstacle] =
                learnopencv::stereo::findLargestObstacle(
                    depth, options.minDepth, options.safeDistance,
                    options.minimumAreaFraction);

            cv::remap(
                leftColor, finalCanvas, maps.leftX, maps.leftY,
                cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
            if (obstacle.has_value()) {
                cv::rectangle(
                    finalCanvas, obstacle->boundingBox,
                    cv::Scalar(0, 0, 255), 3);
                std::ostringstream label;
                label << "WARNING: " << std::fixed
                      << std::setprecision(1)
                      << obstacle->meanDepth << " cm";
                cv::putText(
                    finalCanvas, label.str(),
                    cv::Point(
                        std::max(0, obstacle->boundingBox.x),
                        std::max(30, obstacle->boundingBox.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            } else {
                cv::putText(
                    finalCanvas, "SAFE", cv::Point(30, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 1.4,
                    cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }

            ++frameCount;
            if (options.display) {
                cv::imshow("Obstacle avoidance", finalCanvas);
                cv::imshow(
                    "Disparity",
                    learnopencv::stereo::disparityVisualization(
                        disparity));
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
        if (finalCanvas.empty()) {
            throw std::runtime_error("No stereo frames were processed.");
        }
        if (!options.output.empty()) {
            if (!options.output.parent_path().empty()) {
                std::filesystem::create_directories(
                    options.output.parent_path());
            }
            if (!cv::imwrite(
                    options.output.string(), finalCanvas)) {
                throw std::runtime_error(
                    "Could not write obstacle output.");
            }
            std::cout << "Saved final obstacle view: "
                      << options.output << '\n';
        }
        std::cout << "Processed stereo frames: " << frameCount << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
