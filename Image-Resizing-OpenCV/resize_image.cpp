#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

struct LetterboxResult {
    cv::Mat image;
    double scale;
    cv::Rect content;
};

cv::Mat resizeExact(const cv::Mat& image, int width, int height, int interpolation) {
    if (image.empty()) throw std::invalid_argument("image must be non-empty");
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("target width and height must be positive");
    }
    cv::Mat output;
    cv::resize(image, output, cv::Size(width, height), 0.0, 0.0, interpolation);
    return output;
}

cv::Mat resizeByScale(const cv::Mat& image, double scaleX, double scaleY,
                      int interpolation) {
    if (image.empty()) throw std::invalid_argument("image must be non-empty");
    if (scaleX <= 0.0 || scaleY <= 0.0) {
        throw std::invalid_argument("scale factors must be positive");
    }
    cv::Mat output;
    cv::resize(image, output, cv::Size(), scaleX, scaleY, interpolation);
    return output;
}

LetterboxResult letterbox(const cv::Mat& image, int targetWidth, int targetHeight,
                          const cv::Scalar& color = cv::Scalar(32, 32, 32)) {
    if (image.empty()) throw std::invalid_argument("image must be non-empty");
    if (targetWidth <= 0 || targetHeight <= 0) {
        throw std::invalid_argument("target width and height must be positive");
    }
    const double scale = std::min(
        static_cast<double>(targetWidth) / image.cols,
        static_cast<double>(targetHeight) / image.rows);
    const int contentWidth = std::max(1, static_cast<int>(std::lround(image.cols * scale)));
    const int contentHeight = std::max(1, static_cast<int>(std::lround(image.rows * scale)));
    cv::Mat resized = resizeExact(
        image, contentWidth, contentHeight,
        scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);
    cv::Mat output(targetHeight, targetWidth, image.type(), color);
    const cv::Rect content((targetWidth - contentWidth) / 2,
                           (targetHeight - contentHeight) / 2,
                           contentWidth, contentHeight);
    resized.copyTo(output(content));
    return {output, scale, content};
}

cv::Mat labeledPanel(const cv::Mat& image, const std::string& label) {
    cv::Mat panel = image.clone();
    cv::rectangle(panel, cv::Rect(0, 0, panel.cols, 42), cv::Scalar::all(0), -1);
    cv::putText(panel, label, cv::Point(14, 29), cv::FONT_HERSHEY_SIMPLEX,
                0.72, cv::Scalar::all(255), 2, cv::LINE_AA);
    return panel;
}

cv::Mat makeComparison(const cv::Mat& image) {
    const cv::Size half(image.cols / 2, image.rows / 2);
    cv::Mat areaSmall;
    cv::Mat linearSmall;
    cv::resize(image, areaSmall, half, 0.0, 0.0, cv::INTER_AREA);
    cv::resize(image, linearSmall, half, 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat areaRestored;
    cv::Mat linearRestored;
    cv::resize(areaSmall, areaRestored, image.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(linearSmall, linearRestored, image.size(), 0.0, 0.0, cv::INTER_NEAREST);
    std::vector<cv::Mat> panels{
        labeledPanel(image, "Input"),
        labeledPanel(areaRestored, "Downscale: INTER_AREA"),
        labeledPanel(linearRestored, "Downscale: INTER_LINEAR")};
    cv::Mat comparison;
    cv::hconcat(panels, comparison);
    return comparison;
}

void requireWrite(const fs::path& path, const cv::Mat& image) {
    fs::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("could not write " + path.string());
    }
}

int selfTest() {
    cv::Mat fixture(40, 80, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat exact = resizeExact(fixture, 20, 10, cv::INTER_AREA);
    const cv::Mat scaled = resizeByScale(fixture, 0.5, 0.5, cv::INTER_AREA);
    const LetterboxResult boxed = letterbox(fixture, 100, 100);
    const cv::Mat oddGeometry(4, 5, CV_8UC3, cv::Scalar::all(0));
    const LetterboxResult oddBoxed = letterbox(oddGeometry, 4, 2);
    bool rejectedEmpty = false;
    try {
        static_cast<void>(letterbox(cv::Mat(), 100, 100));
    } catch (const std::invalid_argument&) {
        rejectedEmpty = true;
    }
    if (exact.size() != cv::Size(20, 10) ||
        scaled.size() != cv::Size(40, 20) ||
        boxed.image.size() != cv::Size(100, 100) ||
        boxed.content.size() != cv::Size(100, 50) ||
        boxed.content.y != 25 ||
        oddBoxed.content.size() != cv::Size(3, 2) ||
        !rejectedEmpty) {
        std::cerr << "self-test failed\n";
        return 1;
    }
    std::cout << "self-test passed\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        fs::path input = fs::path(__FILE__).parent_path() / "assets" / "sample-scene.png";
        fs::path outputDir = fs::path(__FILE__).parent_path() / "outputs";
        bool display = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--self-test") return selfTest();
            if (arg == "--display") display = true;
            else if (arg == "--input" && i + 1 < argc) input = argv[++i];
            else if (arg == "--output-dir" && i + 1 < argc) outputDir = argv[++i];
            else if (arg != "--display") throw std::invalid_argument("unknown or incomplete option: " + arg);
        }
        const cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("could not read " + input.string());
        const cv::Mat downArea = resizeExact(
            image, image.cols / 2, image.rows / 2, cv::INTER_AREA);
        const cv::Mat upLinear = resizeByScale(image, 1.5, 1.5, cv::INTER_LINEAR);
        const cv::Mat upCubic = resizeByScale(image, 1.5, 1.5, cv::INTER_CUBIC);
        const LetterboxResult boxed = letterbox(image, 640, 640);
        requireWrite(outputDir / "downscale-inter-area.png", downArea);
        requireWrite(outputDir / "upscale-inter-linear.png", upLinear);
        requireWrite(outputDir / "upscale-inter-cubic.png", upCubic);
        requireWrite(outputDir / "letterbox-640.png", boxed.image);
        const cv::Mat comparison = makeComparison(image);
        requireWrite(outputDir / "resize-comparison.png", comparison);
        std::cout << "input=" << image.cols << "x" << image.rows
                  << " down=" << downArea.cols << "x" << downArea.rows
                  << " letterbox_content=" << boxed.content.width << "x"
                  << boxed.content.height << '\n';
        if (display) {
            cv::imshow("Resize interpolation comparison", comparison);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
