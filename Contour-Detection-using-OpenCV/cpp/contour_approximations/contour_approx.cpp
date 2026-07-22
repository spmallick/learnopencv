#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
    fs::path image1 = fs::path(TUTORIAL_INPUT_DIR) / "image_1.jpg";
    fs::path image2 = fs::path(TUTORIAL_INPUT_DIR) / "image_2.jpg";
    fs::path output_dir = fs::current_path();
    bool display = true;
    bool validate = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "--image1" || argument == "--image2" ||
             argument == "--output-dir") && i + 1 >= argc) {
            throw std::invalid_argument(argument + " requires a path");
        }
        if (argument == "--image1") {
            options.image1 = argv[++i];
        } else if (argument == "--image2") {
            options.image2 = argv[++i];
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

cv::Mat read_image(const fs::path& path) {
    cv::Mat image = cv::imread(path.string());
    if (image.empty()) {
        throw std::runtime_error("Could not read input image: " + path.string());
    }
    return image;
}

void show_image(const std::string& title, const cv::Mat& image, bool display) {
    if (display) {
        cv::imshow(title, image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Could not write output image: " + path.string());
    }
}

std::vector<std::vector<cv::Point>> find_binary_contours(
    const cv::Mat& image, int method, cv::Mat* threshold_output = nullptr
) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat threshold_image;
    cv::threshold(gray, threshold_image, 150, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(threshold_image.clone(), contours, hierarchy, cv::RETR_TREE, method);
    if (threshold_output != nullptr) {
        *threshold_output = threshold_image;
    }
    return contours;
}

std::size_t point_count(const std::vector<std::vector<cv::Point>>& contours) {
    std::size_t total = 0;
    for (const auto& contour : contours) {
        total += contour.size();
    }
    return total;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        fs::create_directories(options.output_dir);

        const cv::Mat image1 = read_image(options.image1);
        cv::Mat threshold_image;
        const auto contours_none =
            find_binary_contours(image1, cv::CHAIN_APPROX_NONE, &threshold_image);
        const auto contours_simple = find_binary_contours(image1, cv::CHAIN_APPROX_SIMPLE);

        show_image("Binary image", threshold_image, options.display);
        write_image(options.output_dir / "image_thres1.jpg", threshold_image);

        cv::Mat none_rendered = image1.clone();
        cv::drawContours(
            none_rendered, contours_none, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA
        );
        show_image("None approximation", none_rendered, options.display);
        write_image(options.output_dir / "contours_none_image1.jpg", none_rendered);

        cv::Mat simple_rendered = image1.clone();
        cv::drawContours(
            simple_rendered, contours_simple, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA
        );
        show_image("Simple approximation", simple_rendered, options.display);
        write_image(options.output_dir / "contours_simple_image1.jpg", simple_rendered);

        const cv::Mat image2 = read_image(options.image2);
        const auto image2_contours = find_binary_contours(image2, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat image2_rendered = image2.clone();
        cv::drawContours(
            image2_rendered, image2_contours, -1, cv::Scalar(0, 255, 0), 2,
            cv::LINE_AA
        );
        show_image("SIMPLE approximation contours", image2_rendered, options.display);
        write_image(options.output_dir / "contours_simple_image2.jpg", image2_rendered);

        cv::Mat points_only = image2.clone();
        for (const auto& contour : image2_contours) {
            for (const cv::Point& point : contour) {
                cv::circle(points_only, point, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }
        show_image("CHAIN_APPROX_SIMPLE points", points_only, options.display);
        write_image(options.output_dir / "contour_point_simple.jpg", points_only);

        const std::size_t none_points = point_count(contours_none);
        const std::size_t simple_points = point_count(contours_simple);
        std::cout << "none_contours=" << contours_none.size()
                  << ", none_points=" << none_points
                  << ", simple_contours=" << contours_simple.size()
                  << ", simple_points=" << simple_points
                  << ", image2_simple_contours=" << image2_contours.size()
                  << ", image2_simple_points=" << point_count(image2_contours) << '\n';

        if (options.validate) {
            if (contours_none.size() != contours_simple.size() ||
                simple_points >= none_points || image2_contours.empty()) {
                throw std::runtime_error("Contour approximation validation failed");
            }
            std::cout << "Validation passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
