#include "color_spaces.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef COLOR_SPACES_PROJECT_DIR
#error "COLOR_SPACES_PROJECT_DIR must be defined by CMake"
#endif

namespace {

constexpr int kPanelWidth = 420;
const std::string kWindowName = "Move the mouse over the image; press Esc to exit";

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
    std::optional<std::filesystem::path> output;
    std::optional<int> x;
    std::optional<int> y;
    bool noDisplay = false;
    bool validate = false;
    bool help = false;
};

Options parseArguments(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--input") {
            if (++index >= argc) {
                throw std::invalid_argument("--input requires a path");
            }
            options.input = std::filesystem::absolute(argv[index]);
        } else if (argument == "--output") {
            if (++index >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            options.output = std::filesystem::absolute(argv[index]);
        } else if (argument == "--x") {
            if (++index >= argc) {
                throw std::invalid_argument("--x requires an integer");
            }
            options.x = parseInteger(argv[index]);
        } else if (argument == "--y") {
            if (++index >= argc) {
                throw std::invalid_argument("--y requires an integer");
            }
            options.y = parseInteger(argv[index]);
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

cv::Mat renderPixelPanel(const cv::Mat& image, const int x, const int y) {
    if (x < 0 || x >= image.cols || y < 0 || y >= image.rows) {
        throw std::invalid_argument(
            "pixel coordinate is outside image bounds " +
            std::to_string(image.cols) + "x" + std::to_string(image.rows));
    }

    const color_spaces::PixelValues values =
        color_spaces::convertPixel(image.at<cv::Vec3b>(y, x));
    cv::Mat panel = cv::Mat::zeros(image.rows, kPanelWidth, CV_8UC3);
    const std::array<std::string, 5> lines = {
        "Pixel (x=" + std::to_string(x) + ", y=" + std::to_string(y) + ")",
        "BGR   " + color_spaces::vectorText(values.bgr),
        "HSV   " + color_spaces::vectorText(values.hsv),
        "YCrCb " + color_spaces::vectorText(values.yCrCb),
        "Lab   " + color_spaces::vectorText(values.lab),
    };

    for (std::size_t index = 0; index < lines.size(); ++index) {
        cv::putText(
            panel,
            lines[index],
            cv::Point(20, 45 + static_cast<int>(index) * 55),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA);
    }

    cv::Mat marked = image.clone();
    cv::drawMarker(
        marked,
        cv::Point(x, y),
        cv::Scalar(0, 255, 255),
        cv::MARKER_CROSS,
        18,
        2);
    cv::Mat combined;
    cv::hconcat(marked, panel, combined);
    return combined;
}

void validateKnownPixel() {
    const color_spaces::PixelValues values =
        color_spaces::convertPixel(cv::Vec3b(40, 158, 16));
    if (values.bgr != cv::Vec3b(40, 158, 16) ||
        values.hsv != cv::Vec3b(65, 229, 158) ||
        values.yCrCb != cv::Vec3b(102, 67, 93) ||
        values.lab != cv::Vec3b(145, 71, 177)) {
        throw std::runtime_error("known pixel conversion regression");
    }
}

void runValidation(const Options& options) {
    validateKnownPixel();
    const cv::Mat image = color_spaces::readBgr(options.input);
    const cv::Mat rendered =
        renderPixelPanel(image, image.cols / 2, image.rows / 2);
    if (rendered.rows != image.rows || rendered.cols != image.cols + kPanelWidth) {
        throw std::runtime_error("rendered panel dimensions are incorrect");
    }
    if (options.output.has_value()) {
        color_spaces::writeImage(*options.output, rendered);
    }
    std::cout << "VALIDATION PASSED: image=" << image.cols << 'x' << image.rows
              << ", panel=" << rendered.cols << 'x' << rendered.rows << '\n';
}

struct MouseState {
    const cv::Mat* image = nullptr;
    cv::Mat rendered;
};

void onMouse(const int event, const int x, const int y, int, void* userData) {
    if (event != cv::EVENT_MOUSEMOVE || userData == nullptr) {
        return;
    }
    auto* state = static_cast<MouseState*>(userData);
    if (state->image != nullptr &&
        x >= 0 && x < state->image->cols &&
        y >= 0 && y < state->image->rows) {
        state->rendered = renderPixelPanel(*state->image, x, y);
        cv::imshow(kWindowName, state->rendered);
    }
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--input IMAGE] [--x X --y Y] [--output IMAGE]"
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

        const cv::Mat image = color_spaces::readBgr(options.input);
        const int x = options.x.value_or(image.cols / 2);
        const int y = options.y.value_or(image.rows / 2);
        cv::Mat rendered = renderPixelPanel(image, x, y);
        const color_spaces::PixelValues values =
            color_spaces::convertPixel(image.at<cv::Vec3b>(y, x));

        std::cout << "Input: " << options.input << '\n';
        std::cout << "Pixel: x=" << x << ", y=" << y << '\n';
        std::cout << "BGR: " << color_spaces::vectorText(values.bgr) << '\n';
        std::cout << "HSV: " << color_spaces::vectorText(values.hsv) << '\n';
        std::cout << "YCrCb: " << color_spaces::vectorText(values.yCrCb) << '\n';
        std::cout << "Lab: " << color_spaces::vectorText(values.lab) << '\n';

        if (options.output.has_value()) {
            color_spaces::writeImage(*options.output, rendered);
            std::cout << "Wrote: " << *options.output << '\n';
        }
        if (options.noDisplay) {
            return 0;
        }

        MouseState state{&image, rendered};
        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(kWindowName, onMouse, &state);
        cv::imshow(kWindowName, state.rendered);
        while ((cv::waitKey(20) & 0xFF) != 27) {
        }
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        printUsage(argv[0]);
        return 2;
    }
}
