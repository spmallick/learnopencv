// This code is written by Sunita Nayak at BigVision LLC.
// It is subject to the license terms in the LICENSE file in this folder.

#include "colorization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const colorization::CommonOptions options = colorization::parseOptions(
            argc,
            argv,
            "greyscaleImage.png",
            "colorized-image.png",
            false);

        const cv::Mat input = cv::imread(options.input.string(), cv::IMREAD_COLOR);
        if (input.empty()) {
            throw std::runtime_error(
                "Could not read input image: " + options.input.string());
        }

        cv::dnn::Net network = colorization::loadNetwork(options.model);
        const auto start = std::chrono::steady_clock::now();
        auto [output, chromaScore] =
            colorization::colorizeFrame(input, network);
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - start;

        const std::filesystem::path parent = options.output.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        if (!cv::imwrite(options.output.string(), output)) {
            throw std::runtime_error(
                "Could not write output image: " + options.output.string());
        }
        if (options.validate) {
            colorization::validateOutput(input, output, chromaScore);
        }

        std::cout << "Saved " << options.output << '\n'
                  << "Inference time: " << std::fixed << std::setprecision(3)
                  << elapsed.count() << " seconds\n"
                  << "Mean predicted chroma: " << chromaScore << '\n';

        if (!options.noDisplay) {
            cv::Mat comparison;
            cv::hconcat(input, output, comparison);
            cv::imshow("Input | Colorized", comparison);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
