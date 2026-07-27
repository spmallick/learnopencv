#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "stereo_depth.hpp"

namespace {

constexpr const char* kWindow = "Depth calibration disparity";

struct Options {
    int leftCamera = 2;
    int rightCamera = 0;
    double maxDistance = 230.0;
    double minDistance = 50.0;
    double sampleStep = 40.0;
    std::filesystem::path maps =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "stereo_rectify_maps.xml";
    std::filesystem::path config =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "depth_estimation_params_cpp.xml";
    std::filesystem::path outputConfig =
        std::filesystem::path(STEREO_SOURCE_DIR) / "data" /
        "depth_estimation_params_cpp_updated.xml";
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
        } else if (argument == "--output-config") {
            options.outputConfig = requireValue("--output-config");
        } else if (argument == "--max-distance") {
            options.maxDistance =
                std::stod(requireValue("--max-distance"));
        } else if (argument == "--min-distance") {
            options.minDistance =
                std::stod(requireValue("--min-distance"));
        } else if (argument == "--sample-step") {
            options.sampleStep =
                std::stod(requireValue("--sample-step"));
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }
    if (options.leftCamera == options.rightCamera) {
        throw std::invalid_argument(
            "Left and right camera indices must be different.");
    }
    if (
        options.minDistance <= 0.0 ||
        options.maxDistance <= options.minDistance ||
        options.sampleStep <= 0.0) {
        throw std::invalid_argument(
            "Distance limits and sample step are invalid.");
    }
    return options;
}

struct CalibrationState {
    cv::Mat disparity;
    double targetDepth = 0.0;
    double sampleStep = 0.0;
    int minDisparity = 0;
    std::vector<double> disparities;
    std::vector<double> depths;
};

void onMouse(int event, int x, int y, int, void* userData) {
    if (event != cv::EVENT_LBUTTONDOWN || userData == nullptr) {
        return;
    }
    auto& state = *static_cast<CalibrationState*>(userData);
    if (
        state.disparity.empty() || x < 0 || y < 0 ||
        x >= state.disparity.cols || y >= state.disparity.rows) {
        return;
    }
    const float value = state.disparity.at<float>(y, x);
    if (
        !std::isfinite(value) ||
        value <= static_cast<float>(state.minDisparity)) {
        std::cout << "Ignored invalid disparity sample.\n";
        return;
    }
    state.disparities.push_back(value);
    state.depths.push_back(state.targetDepth);
    std::cout << "Depth " << state.targetDepth
              << " cm -> disparity " << value << " px\n";
    state.targetDepth -= state.sampleStep;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        auto config = learnopencv::stereo::loadConfig(options.config);
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

        CalibrationState state;
        state.targetDepth = options.maxDistance;
        state.sampleStep = options.sampleStep;
        state.minDisparity = config.minDisparity;
        cv::namedWindow(kWindow, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindow, 800, 600);
        cv::setMouseCallback(kWindow, onMouse, &state);

        while (state.targetDepth >= options.minDistance) {
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
            state.disparity =
                learnopencv::stereo::computeDisparity(
                    leftRectified, rightRectified, config, matcher);
            cv::Mat view =
                learnopencv::stereo::disparityVisualization(
                    state.disparity);
            cv::putText(
                view,
                "Place target at " +
                    std::to_string(
                        static_cast<int>(state.targetDepth)) +
                    " cm and click",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255), 2, cv::LINE_AA);
            cv::imshow(kWindow, view);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        cv::destroyAllWindows();

        const auto fit = learnopencv::stereo::fitDepthModel(
            state.disparities, state.depths, config.minDisparity);
        config.depthScale = fit.scale;
        config.depthOffset = fit.offset;
        config.validate();
        learnopencv::stereo::saveConfig(
            config, options.outputConfig);
        std::cout << "Depth scale: " << fit.scale << '\n'
                  << "Depth offset: " << fit.offset << '\n'
                  << "Calibration RMSE: " << fit.rmse << " cm\n"
                  << "Saved calibrated configuration: "
                  << options.outputConfig << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
