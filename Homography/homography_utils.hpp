#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#include <opencv2/geometry/3d.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef HOMOGRAPHY_ASSET_DIR
#define HOMOGRAPHY_ASSET_DIR "."
#endif

namespace homography_example {

class Arguments {
public:
    Arguments(int argc, char** argv) {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--display" || argument == "--help" ||
                argument == "-h") {
                flags_.insert(argument == "-h" ? "--help" : argument);
                continue;
            }
            if (argument.rfind("--", 0) != 0) {
                throw std::invalid_argument(
                    "Unexpected positional argument: " + argument);
            }
            if (++index >= argc) {
                throw std::invalid_argument(
                    "Missing value for " + argument);
            }
            values_[argument] = argv[index];
        }
    }

    void validate(
        std::initializer_list<std::string> value_options,
        std::initializer_list<std::string> flag_options) const {
        const std::set<std::string> allowed_values(value_options);
        const std::set<std::string> allowed_flags(flag_options);
        for (const auto& [name, value] : values_) {
            static_cast<void>(value);
            if (allowed_values.count(name) == 0) {
                throw std::invalid_argument("Unknown option: " + name);
            }
        }
        for (const auto& name : flags_) {
            if (allowed_flags.count(name) == 0) {
                throw std::invalid_argument("Unknown flag: " + name);
            }
        }
    }

    bool has(const std::string& name) const {
        return values_.count(name) != 0 || flags_.count(name) != 0;
    }

    std::string value(
        const std::string& name, const std::string& fallback) const {
        const auto found = values_.find(name);
        return found == values_.end() ? fallback : found->second;
    }

    int integer(const std::string& name, int fallback) const {
        const auto found = values_.find(name);
        if (found == values_.end()) {
            return fallback;
        }
        std::size_t parsed = 0;
        const int number = std::stoi(found->second, &parsed);
        if (parsed != found->second.size()) {
            throw std::invalid_argument(
                name + " must be an integer, not " + found->second);
        }
        return number;
    }

private:
    std::map<std::string, std::string> values_;
    std::set<std::string> flags_;
};

inline std::filesystem::path assetPath(const std::string& filename) {
    return std::filesystem::path(HOMOGRAPHY_ASSET_DIR) / filename;
}

inline cv::Mat readImage(const std::filesystem::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + path.string());
    }
    return image;
}

inline void writeImage(
    const std::filesystem::path& path, const cv::Mat& image) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Could not write image: " + path.string());
    }
}

inline std::vector<cv::Point2f> validateQuad(
    const std::vector<cv::Point2f>& points,
    const cv::Size& bounds = {},
    const std::string& name = "points") {
    if (points.size() != 4) {
        throw std::invalid_argument(
            name + " must contain exactly four points.");
    }
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument(
                name + " must contain only finite coordinates.");
        }
        if (
            bounds.width > 0 &&
            (point.x < 0 || point.x >= bounds.width || point.y < 0 ||
             point.y >= bounds.height)) {
            throw std::invalid_argument(
                name + " must lie inside the corresponding image.");
        }
    }
    if (std::abs(cv::contourArea(points)) < 1.0) {
        throw std::invalid_argument(
            name + " must enclose a non-zero area.");
    }
    if (!cv::isContourConvex(points)) {
        throw std::invalid_argument(
            name + " must be ordered around a convex quadrilateral.");
    }
    return points;
}

inline std::vector<cv::Point2f> parsePoints(const std::string& text) {
    std::vector<cv::Point2f> points;
    std::stringstream point_stream(text);
    std::string pair;
    while (std::getline(point_stream, pair, ';')) {
        const auto separator = pair.find(',');
        if (separator == std::string::npos ||
            pair.find(',', separator + 1) != std::string::npos) {
            throw std::invalid_argument(
                "Points must use x,y;x,y;x,y;x,y format.");
        }
        try {
            std::size_t x_parsed = 0;
            std::size_t y_parsed = 0;
            const std::string x_text = pair.substr(0, separator);
            const std::string y_text = pair.substr(separator + 1);
            const float x = std::stof(x_text, &x_parsed);
            const float y = std::stof(y_text, &y_parsed);
            if (x_parsed != x_text.size() || y_parsed != y_text.size()) {
                throw std::invalid_argument("Trailing coordinate text.");
            }
            points.emplace_back(x, y);
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "Points must use numeric x,y;x,y;x,y;x,y format.");
        }
    }
    return validateQuad(points);
}

inline std::vector<cv::Point2f> rectangleCorners(
    int width, int height) {
    if (width < 2 || height < 2) {
        throw std::invalid_argument(
            "Width and height must both be at least 2 pixels.");
    }
    return {
        {0.0F, 0.0F},
        {static_cast<float>(width - 1), 0.0F},
        {
            static_cast<float>(width - 1),
            static_cast<float>(height - 1),
        },
        {0.0F, static_cast<float>(height - 1)},
    };
}

inline std::vector<cv::Point2f> imageCorners(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("The image must not be empty.");
    }
    return rectangleCorners(image.cols, image.rows);
}

inline cv::Mat computeHomography(
    const std::vector<cv::Point2f>& source_points,
    const std::vector<cv::Point2f>& destination_points) {
    const auto source = validateQuad(source_points, {}, "source_points");
    const auto destination =
        validateQuad(destination_points, {}, "destination_points");
    cv::Mat homography = cv::findHomography(source, destination, 0);
    if (homography.empty()) {
        throw std::runtime_error("OpenCV could not compute a homography.");
    }
    return homography;
}

inline std::pair<cv::Mat, cv::Mat> rectifyImage(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& source_points,
    int width,
    int height) {
    const auto source =
        validateQuad(source_points, image.size(), "source_points");
    const cv::Mat homography =
        computeHomography(source, rectangleCorners(width, height));
    cv::Mat rectified;
    cv::warpPerspective(
        image, rectified, homography, cv::Size(width, height));
    return {rectified, homography};
}

inline std::pair<cv::Mat, cv::Mat> compositeOnQuad(
    const cv::Mat& source,
    const cv::Mat& destination,
    const std::vector<cv::Point2f>& destination_points) {
    const auto destination_quad = validateQuad(
        destination_points, destination.size(), "destination_points");
    const cv::Mat homography =
        computeHomography(imageCorners(source), destination_quad);

    cv::Mat warped_source;
    cv::warpPerspective(
        source, warped_source, homography, destination.size());

    cv::Mat source_mask(source.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat warped_mask;
    cv::warpPerspective(
        source_mask,
        warped_mask,
        homography,
        destination.size(),
        cv::INTER_NEAREST);

    cv::Mat result = destination.clone();
    warped_source.copyTo(result, warped_mask);
    return {result, homography};
}

struct SelectionData {
    cv::Mat original;
    std::vector<cv::Point2f> points;
};

inline void mouseHandler(
    int event, int x, int y, int flags, void* raw_data) {
    static_cast<void>(flags);
    auto* data = static_cast<SelectionData*>(raw_data);
    if (event == cv::EVENT_LBUTTONDOWN && data->points.size() < 4) {
        data->points.emplace_back(
            static_cast<float>(x), static_cast<float>(y));
    }
}

inline std::vector<cv::Point2f> collectFourPoints(
    const cv::Mat& image,
    const std::string& window_name = "Select four points") {
    if (image.empty()) {
        throw std::invalid_argument("The image must not be empty.");
    }
    SelectionData data{image.clone(), {}};
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window_name, mouseHandler, &data);

    while (true) {
        cv::Mat preview = data.original.clone();
        for (const auto& point : data.points) {
            cv::circle(
                preview, point, 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        }
        cv::putText(
            preview,
            "Points: " + std::to_string(data.points.size()) +
                "/4 | Enter: accept | R: reset | Esc: cancel",
            {12, 28},
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA);
        cv::imshow(window_name, preview);
        const int key = cv::waitKey(20) & 0xFF;
        if ((key == 10 || key == 13 || key == 32) &&
            data.points.size() == 4) {
            break;
        }
        if (key == 'r' || key == 'R') {
            data.points.clear();
        }
        if (key == 27) {
            cv::destroyWindow(window_name);
            throw std::runtime_error("Point selection was cancelled.");
        }
    }
    cv::destroyWindow(window_name);
    return validateQuad(data.points, image.size(), "selected points");
}

inline void showImages(
    const std::vector<std::pair<std::string, cv::Mat>>& images) {
    for (const auto& [name, image] : images) {
        cv::imshow(name, image);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

inline void printRectificationUsage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " [--input IMAGE] [--output IMAGE]\n"
        << "       [--points \"x,y;x,y;x,y;x,y\"] [--width N]"
        << " [--height N] [--display]\n";
}

inline int runRectification(
    int argc, char** argv, const std::string& default_output) {
    const Arguments arguments(argc, argv);
    arguments.validate(
        {"--input", "--output", "--points", "--width", "--height"},
        {"--display", "--help"});
    if (arguments.has("--help")) {
        printRectificationUsage(argv[0]);
        return 0;
    }

    const std::filesystem::path input = arguments.value(
        "--input", assetPath("book1.jpg").string());
    const std::filesystem::path output =
        arguments.value("--output", default_output);
    const int width = arguments.integer("--width", 300);
    const int height = arguments.integer("--height", 400);
    const cv::Mat source = readImage(input);

    std::vector<cv::Point2f> source_points;
    if (arguments.has("--points")) {
        source_points = parsePoints(arguments.value("--points", ""));
    } else {
        std::cout
            << "Click four corners clockwise from the top-left. "
            << "Press Enter after the fourth point.\n";
        source_points = collectFourPoints(source);
    }

    const auto [result, homography] =
        rectifyImage(source, source_points, width, height);
    static_cast<void>(homography);
    writeImage(output, result);
    std::cout << "Saved " << width << 'x' << height
              << " rectified image to "
              << std::filesystem::absolute(output) << '\n';

    if (arguments.has("--display")) {
        showImages({{"Source", source}, {"Rectified", result}});
    }
    return 0;
}

inline void printCompositeUsage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " [--source IMAGE] [--destination IMAGE] [--output IMAGE]\n"
        << "       [--points \"x,y;x,y;x,y;x,y\"] [--display]\n";
}

inline int runComposite(
    int argc, char** argv, const std::string& default_output) {
    const Arguments arguments(argc, argv);
    arguments.validate(
        {"--source", "--destination", "--output", "--points"},
        {"--display", "--help"});
    if (arguments.has("--help")) {
        printCompositeUsage(argv[0]);
        return 0;
    }

    const std::filesystem::path source_path = arguments.value(
        "--source", assetPath("first-image.jpg").string());
    const std::filesystem::path destination_path = arguments.value(
        "--destination", assetPath("times-square.jpg").string());
    const std::filesystem::path output =
        arguments.value("--output", default_output);
    const cv::Mat source = readImage(source_path);
    const cv::Mat destination = readImage(destination_path);

    std::vector<cv::Point2f> destination_points;
    if (arguments.has("--points")) {
        destination_points =
            parsePoints(arguments.value("--points", ""));
    } else {
        std::cout
            << "Click four destination corners clockwise from the top-left. "
            << "Press Enter after the fourth point.\n";
        destination_points = collectFourPoints(destination);
    }

    const auto [result, homography] =
        compositeOnQuad(source, destination, destination_points);
    static_cast<void>(homography);
    writeImage(output, result);
    std::cout << "Saved composite image to "
              << std::filesystem::absolute(output) << '\n';

    if (arguments.has("--display")) {
        showImages(
            {
                {"Source", source},
                {"Destination", destination},
                {"Composite", result},
            });
    }
    return 0;
}

}  // namespace homography_example
