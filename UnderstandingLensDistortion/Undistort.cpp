#include "lens_calibration.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef LENS_SOURCE_DIR
#define LENS_SOURCE_DIR "."
#endif

namespace {

struct Options {
    std::filesystem::path imagesDirectory =
        std::filesystem::path(LENS_SOURCE_DIR) / "images";
    std::filesystem::path outputDirectory =
        std::filesystem::path(LENS_SOURCE_DIR) / "outputs";
    int boardColumns = 6;
    int boardRows = 9;
    double squareSize = 1.0;
    double alpha = 1.0;
    bool crop = false;
    bool requireAll = false;
    bool show = false;
};

std::string requireValue(
    const int argc,
    char** argv,
    int& index,
    const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value.");
    }
    return argv[++index];
}

Options parseOptions(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--images-dir") {
            options.imagesDirectory =
                requireValue(argc, argv, index, argument);
        } else if (argument == "--output-dir") {
            options.outputDirectory =
                requireValue(argc, argv, index, argument);
        } else if (argument == "--board-columns") {
            options.boardColumns = std::stoi(
                requireValue(argc, argv, index, argument));
        } else if (argument == "--board-rows") {
            options.boardRows = std::stoi(
                requireValue(argc, argv, index, argument));
        } else if (argument == "--square-size") {
            options.squareSize = std::stod(
                requireValue(argc, argv, index, argument));
        } else if (argument == "--alpha") {
            options.alpha = std::stod(
                requireValue(argc, argv, index, argument));
        } else if (argument == "--crop") {
            options.crop = true;
        } else if (argument == "--require-all") {
            options.requireAll = true;
        } else if (argument == "--show") {
            options.show = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: Undistort [options]\n"
                << "  --images-dir PATH       Calibration JPEG directory\n"
                << "  --output-dir PATH       Saved-result directory\n"
                << "  --board-columns N       Inner-corner columns (default 6)\n"
                << "  --board-rows N          Inner-corner rows (default 9)\n"
                << "  --square-size VALUE     Checkerboard square size\n"
                << "  --alpha VALUE           Valid-pixel/free-scaling balance\n"
                << "  --crop                  Crop to the valid-pixel ROI\n"
                << "  --require-all           Fail if any detection fails\n"
                << "  --show                  Open result windows\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }
    return options;
}

void writeImage(
    const std::filesystem::path& path,
    const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error(
            "Could not write output image: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const auto paths =
            lens_distortion::discoverImages(options.imagesDirectory);
        const auto calibration = lens_distortion::calibrateFromImages(
            paths,
            cv::Size(options.boardColumns, options.boardRows),
            options.squareSize,
            options.requireAll);

        const cv::Mat sample = cv::imread(
            calibration.successfulImages.front().string(),
            cv::IMREAD_COLOR);
        if (sample.empty()) {
            throw std::runtime_error(
                "Could not reread the first successful image.");
        }
        const auto direct = lens_distortion::undistortImage(
            sample, calibration, options.alpha, "direct", options.crop);
        const auto remapped = lens_distortion::undistortImage(
            sample, calibration, options.alpha, "remap", options.crop);

        std::filesystem::create_directories(options.outputDirectory);
        writeImage(
            options.outputDirectory / "calibration-corners.jpg",
            calibration.cornerPreview);
        writeImage(
            options.outputDirectory / "undistorted-direct.jpg",
            direct.image);
        writeImage(
            options.outputDirectory / "undistorted-remap.jpg",
            remapped.image);
        const auto calibrationPath =
            options.outputDirectory / "calibration.yml";
        lens_distortion::saveCalibration(
            calibration, calibrationPath);

        std::cout
            << "Checkerboards detected: "
            << calibration.successfulImages.size() << "/"
            << paths.size() << "\n"
            << "Image size: " << calibration.imageSize.width << "x"
            << calibration.imageSize.height << "\n"
            << std::setprecision(10)
            << "OpenCV calibration RMS: " << calibration.rms << "\n"
            << "Reprojection RMSE: "
            << calibration.reprojectionRMSE << " px\n"
            << "Alpha-" << options.alpha << " ROI: ("
            << direct.roi.x << ", " << direct.roi.y << ", "
            << direct.roi.width << ", " << direct.roi.height << ")\n"
            << "Saved calibration: " << calibrationPath << "\n"
            << "Saved corrected images under: "
            << options.outputDirectory << "\n";

        if (options.show) {
            cv::imshow(
                "Calibration corners", calibration.cornerPreview);
            cv::imshow("Undistorted image", direct.image);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
