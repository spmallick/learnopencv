#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct Options {
    std::filesystem::path input =
        std::filesystem::path(OTSU_SOURCE_DIR) / "boat.jpg";
    std::filesystem::path output =
        std::filesystem::path(OTSU_SOURCE_DIR) / "outputs" /
        "otsu-opencv-cpp.png";
    bool blur = false;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input" && index + 1 < argc) {
            options.input = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            options.output = argv[++index];
        } else if (argument == "--blur") {
            options.blur = true;
        } else {
            throw std::invalid_argument(
                "Usage: otsu_method [--input IMAGE] [--output IMAGE] "
                "[--blur]");
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        cv::Mat image =
            cv::imread(options.input.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error(
                "Could not read grayscale image: " + options.input.string());
        }

        cv::Mat processed = image;
        if (options.blur) {
            cv::GaussianBlur(image, processed, cv::Size(5, 5), 0.0);
        }

        cv::Mat binary;
        const double selectedThreshold = cv::threshold(
            processed, binary, 0.0, 255.0,
            cv::THRESH_BINARY | cv::THRESH_OTSU);

        if (!options.output.parent_path().empty()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        if (!cv::imwrite(options.output.string(), binary)) {
            throw std::runtime_error(
                "Could not write output image: " + options.output.string());
        }

        std::cout << "OpenCV Otsu threshold: " << selectedThreshold << '\n'
                  << "Saved binary image: " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
