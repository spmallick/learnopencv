#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TUTORIAL_INPUT_DIR
#define TUTORIAL_INPUT_DIR "../../input"
#endif

namespace fs = std::filesystem;

struct Options {
    fs::path input = fs::path(TUTORIAL_INPUT_DIR) / "custom_colors.jpg";
    fs::path output_dir = fs::current_path();
    bool display = true;
    bool validate = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "--input" || argument == "--output-dir") && i + 1 >= argc) {
            throw std::invalid_argument(argument + " requires a path");
        }
        if (argument == "--input") {
            options.input = argv[++i];
        } else if (argument == "--output-dir") {
            options.output_dir = argv[++i];
        } else if (argument == "--no-display") {
            options.display = false;
        } else if (argument == "--validate") {
            options.validate = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    return options;
}

void show_image(const std::string& title, const cv::Mat& image, bool display) {
    if (display) {
        cv::imshow(title, image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const cv::Mat image = cv::imread(options.input.string());
        if (image.empty()) {
            throw std::runtime_error("Could not read input image: " + options.input.string());
        }
        fs::create_directories(options.output_dir);

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Mat threshold_image;
        cv::threshold(gray, threshold_image, 150, 255, cv::THRESH_BINARY);

        struct RetrievalMode {
            const char* name;
            int value;
        };
        const std::array<RetrievalMode, 4> modes{{
            {"list", cv::RETR_LIST},
            {"external", cv::RETR_EXTERNAL},
            {"ccomp", cv::RETR_CCOMP},
            {"tree", cv::RETR_TREE},
        }};
        std::array<std::size_t, 4> counts{};

        for (std::size_t index = 0; index < modes.size(); ++index) {
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(
                threshold_image.clone(), contours, hierarchy, modes[index].value,
                cv::CHAIN_APPROX_NONE
            );

            cv::Mat rendered = image.clone();
            cv::drawContours(
                rendered, contours, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA
            );
            show_image(modes[index].name, rendered, options.display);

            const fs::path output =
                options.output_dir / (std::string("contours_retr_") + modes[index].name + ".jpg");
            if (!cv::imwrite(output.string(), rendered)) {
                throw std::runtime_error("Could not write output image: " + output.string());
            }
            counts[index] = contours.size();
            std::cout << modes[index].name << ": contours=" << contours.size()
                      << ", hierarchy=" << hierarchy.size() << '\n';
        }

        if (options.validate) {
            if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0 || counts[3] == 0 ||
                counts[1] > counts[0] || counts[2] != counts[3]) {
                throw std::runtime_error("Contour retrieval-mode validation failed");
            }
            std::cout << "Validation passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
