#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

struct EdgeResult {
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat sobelX;
    cv::Mat sobelY;
    cv::Mat magnitude;
    cv::Mat canny;
};

cv::Mat normalizeGradient(const cv::Mat& gradient) {
    cv::Mat absoluteGradient;
    cv::absdiff(gradient, cv::Scalar::all(0), absoluteGradient);
    cv::Mat output;
    cv::normalize(absoluteGradient, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    return output;
}

EdgeResult detectEdges(const cv::Mat& imageBgr, int low, int high, int blurSize) {
    if (imageBgr.empty() || imageBgr.type() != CV_8UC3) {
        throw std::invalid_argument("input must be a non-empty 8-bit BGR image");
    }
    if (low < 0 || low >= high || high > 255) {
        throw std::invalid_argument("thresholds must satisfy 0 <= low < high <= 255");
    }
    if (blurSize < 3 || blurSize % 2 == 0) {
        throw std::invalid_argument("blur size must be an odd integer of at least 3");
    }

    EdgeResult result;
    cv::cvtColor(imageBgr, result.gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(result.gray, result.blurred, cv::Size(blurSize, blurSize), 0);

    cv::Mat sobelX32f;
    cv::Mat sobelY32f;
    cv::Sobel(result.blurred, sobelX32f, CV_32F, 1, 0, 3);
    cv::Sobel(result.blurred, sobelY32f, CV_32F, 0, 1, 3);
    cv::Mat magnitude32f;
    cv::magnitude(sobelX32f, sobelY32f, magnitude32f);

    result.sobelX = normalizeGradient(sobelX32f);
    result.sobelY = normalizeGradient(sobelY32f);
    result.magnitude = normalizeGradient(magnitude32f);
    cv::Canny(result.blurred, result.canny, low, high, 3, true);
    return result;
}

cv::Mat makeComparison(const cv::Mat& imageBgr, const EdgeResult& result) {
    std::vector<cv::Mat> panels;
    panels.push_back(imageBgr.clone());
    cv::Mat magnitudeBgr;
    cv::Mat cannyBgr;
    cv::cvtColor(result.magnitude, magnitudeBgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(result.canny, cannyBgr, cv::COLOR_GRAY2BGR);
    panels.push_back(magnitudeBgr);
    panels.push_back(cannyBgr);
    const std::vector<std::string> labels{"Input", "Sobel magnitude", "Canny"};
    for (std::size_t i = 0; i < panels.size(); ++i) {
        cv::rectangle(panels[i], cv::Rect(0, 0, panels[i].cols, 42), cv::Scalar::all(0), -1);
        cv::putText(panels[i], labels[i], cv::Point(14, 29), cv::FONT_HERSHEY_SIMPLEX,
                    0.72, cv::Scalar::all(255), 2, cv::LINE_AA);
    }
    cv::Mat comparison;
    cv::hconcat(panels, comparison);
    return comparison;
}

void requireWrite(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("could not write " + path.string());
    }
}

int selfTest() {
    cv::Mat fixture = cv::Mat::zeros(96, 96, CV_8UC3);
    cv::rectangle(fixture, cv::Rect(20, 24, 50, 44), cv::Scalar::all(255), -1);
    const EdgeResult result = detectEdges(fixture, 50, 150, 5);
    const int edgePixels = cv::countNonZero(result.canny);
    if (result.canny.type() != CV_8UC1 || edgePixels < 150 || edgePixels > 250 ||
        cv::countNonZero(result.magnitude) == 0) {
        std::cerr << "self-test failed: edge_pixels=" << edgePixels << '\n';
        return 1;
    }
    std::cout << "self-test passed: edge_pixels=" << edgePixels << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        fs::path input = fs::path(__FILE__).parent_path() / "assets" / "sample-scene.png";
        fs::path outputDir = fs::path(__FILE__).parent_path() / "outputs";
        int low = 100;
        int high = 200;
        int blurSize = 5;
        bool display = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--self-test") return selfTest();
            if (arg == "--display") display = true;
            else if (arg == "--input" && i + 1 < argc) input = argv[++i];
            else if (arg == "--output-dir" && i + 1 < argc) outputDir = argv[++i];
            else if (arg == "--low" && i + 1 < argc) low = std::stoi(argv[++i]);
            else if (arg == "--high" && i + 1 < argc) high = std::stoi(argv[++i]);
            else if (arg == "--blur-size" && i + 1 < argc) blurSize = std::stoi(argv[++i]);
            else if (arg != "--display") throw std::invalid_argument("unknown or incomplete option: " + arg);
        }

        const cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("could not read " + input.string());
        const EdgeResult result = detectEdges(image, low, high, blurSize);
        fs::create_directories(outputDir);
        requireWrite(outputDir / "gray.png", result.gray);
        requireWrite(outputDir / "sobel-x.png", result.sobelX);
        requireWrite(outputDir / "sobel-y.png", result.sobelY);
        requireWrite(outputDir / "sobel-magnitude.png", result.magnitude);
        requireWrite(outputDir / "canny.png", result.canny);
        const cv::Mat comparison = makeComparison(image, result);
        requireWrite(outputDir / "edge-comparison.png", comparison);
        std::cout << "size=" << image.cols << "x" << image.rows
                  << " canny_pixels=" << cv::countNonZero(result.canny)
                  << " outputs=6\n";
        if (display) {
            cv::imshow("Sobel and Canny edge detection", comparison);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
