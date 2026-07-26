/**
 * OpenCV SimpleBlobDetector example.
 *
 * Copyright 2015 by Satya Mallick <spmallick@gmail.com>
 */

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#else
#include <opencv2/features2d.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

struct Options {
    std::filesystem::path input;
    std::filesystem::path output{"blob-keypoints.png"};
    bool display{false};
};

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable
              << " --input IMAGE [--output IMAGE] [--display]\n";
}

Options parseArguments(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input" || argument == "--output") {
            if (++index >= argc) {
                throw std::invalid_argument("Missing value for " + argument);
            }
            const std::filesystem::path value = argv[index];
            if (argument == "--input") {
                options.input = value;
            } else {
                options.output = value;
            }
        } else if (argument == "--display") {
            options.display = true;
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    if (options.input.empty()) {
        throw std::invalid_argument("--input is required.");
    }
    return options;
}

cv::Ptr<cv::SimpleBlobDetector> createDetector() {
    cv::SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 200;
    params.thresholdStep = 10;
    params.minRepeatability = 2;
    params.minDistBetweenBlobs = 10;

    params.filterByColor = true;
    params.blobColor = 0;

    params.filterByArea = true;
    params.minArea = 1500;
    params.maxArea = 5000;

    params.filterByCircularity = true;
    params.minCircularity = 0.1F;
    params.maxCircularity = 1.0F;

    params.filterByConvexity = true;
    params.minConvexity = 0.87F;
    params.maxConvexity = 1.0F;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.01F;
    params.maxInertiaRatio = 1.0F;

    return cv::SimpleBlobDetector::create(params);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        const cv::Mat image =
            cv::imread(options.input.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error(
                "Could not read input image: " + options.input.string());
        }

        std::vector<cv::KeyPoint> keypoints;
        createDetector()->detect(image, keypoints);

        cv::Mat visualization;
        cv::drawKeypoints(
            image,
            keypoints,
            visualization,
            cv::Scalar(0, 0, 255),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (!options.output.parent_path().empty()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        if (!cv::imwrite(options.output.string(), visualization)) {
            throw std::runtime_error(
                "Could not write output image: " + options.output.string());
        }

        std::cout << "Detected " << keypoints.size() << " blobs.\n"
                  << "Saved visualization to "
                  << std::filesystem::absolute(options.output) << '\n';

        if (options.display) {
            cv::imshow("Detected blobs", visualization);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
