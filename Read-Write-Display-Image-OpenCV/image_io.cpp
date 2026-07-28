#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

cv::Mat readImage(const fs::path& path, int flags) {
    const cv::Mat image = cv::imread(path.string(), flags);
    if (image.empty()) throw std::runtime_error("could not decode " + path.string());
    return image;
}

void writeImage(const fs::path& path, const cv::Mat& image,
                const std::vector<int>& parameters = {}) {
    if (image.empty()) throw std::invalid_argument("refusing to write an empty image");
    fs::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(), image, parameters)) {
        throw std::runtime_error("could not encode " + path.string());
    }
}

cv::Mat makeComparison(const cv::Mat& color, const cv::Mat& gray) {
    cv::Mat grayBgr;
    cv::cvtColor(gray, grayBgr, cv::COLOR_GRAY2BGR);
    std::vector<cv::Mat> panels{color.clone(), grayBgr};
    const std::vector<std::string> labels{"IMREAD_COLOR", "IMREAD_GRAYSCALE"};
    for (std::size_t i = 0; i < panels.size(); ++i) {
        cv::rectangle(panels[i], cv::Rect(0, 0, panels[i].cols, 42), cv::Scalar::all(0), -1);
        cv::putText(panels[i], labels[i], cv::Point(14, 29), cv::FONT_HERSHEY_SIMPLEX,
                    0.72, cv::Scalar::all(255), 2, cv::LINE_AA);
    }
    cv::Mat comparison;
    cv::hconcat(panels, comparison);
    return comparison;
}

double runImageIo(const fs::path& input, const fs::path& outputDir) {
    const cv::Mat color = readImage(input, cv::IMREAD_COLOR);
    const cv::Mat gray = readImage(input, cv::IMREAD_GRAYSCALE);
    const cv::Mat unchanged = readImage(input, cv::IMREAD_UNCHANGED);

    const fs::path pngPath = outputDir / "lossless-copy.png";
    const fs::path jpegPath = outputDir / "quality-90.jpg";
    writeImage(pngPath, color, {cv::IMWRITE_PNG_COMPRESSION, 6});
    writeImage(outputDir / "grayscale.png", gray);
    writeImage(jpegPath, color, {cv::IMWRITE_JPEG_QUALITY, 90});
    writeImage(outputDir / "image-io-comparison.png", makeComparison(color, gray));

    const cv::Mat pngRoundtrip = readImage(pngPath, cv::IMREAD_COLOR);
    cv::Mat pngDifference;
    cv::absdiff(color, pngRoundtrip, pngDifference);
    if (cv::countNonZero(pngDifference.reshape(1)) != 0) {
        throw std::runtime_error("PNG round trip unexpectedly changed pixel values");
    }

    const cv::Mat jpegRoundtrip = readImage(jpegPath, cv::IMREAD_COLOR);
    cv::Mat jpegDifference;
    cv::absdiff(color, jpegRoundtrip, jpegDifference);
    const double jpegMae = cv::mean(jpegDifference)[0] / 3.0 +
                           cv::mean(jpegDifference)[1] / 3.0 +
                           cv::mean(jpegDifference)[2] / 3.0;
    std::cout << "size=" << color.cols << "x" << color.rows
              << " channels=" << color.channels()
              << " unchanged_channels=" << unchanged.channels()
              << " jpeg_mae=" << jpegMae << '\n';
    return jpegMae;
}

int selfTest() {
    const fs::path directory = fs::current_path() / "image-io-self-test";
    const fs::path input = directory / "fixture.png";
    cv::Mat fixture(24, 32, CV_8UC3);
    for (int row = 0; row < fixture.rows; ++row) {
        for (int col = 0; col < fixture.cols; ++col) {
            fixture.at<cv::Vec3b>(row, col) =
                cv::Vec3b(static_cast<unsigned char>(col * 7),
                          static_cast<unsigned char>(row * 9),
                          static_cast<unsigned char>((row + col) * 4));
        }
    }
    writeImage(input, fixture);
    const double jpegMae = runImageIo(input, directory / "outputs");
    const bool passed = jpegMae >= 0.0 && jpegMae < 15.0;
    fs::remove_all(directory);
    if (!passed) {
        std::cerr << "self-test failed: jpeg_mae=" << jpegMae << '\n';
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
        runImageIo(input, outputDir);
        if (display) {
            const cv::Mat color = readImage(input, cv::IMREAD_COLOR);
            const cv::Mat gray = readImage(input, cv::IMREAD_GRAYSCALE);
            cv::imshow("Read and display images", makeComparison(color, gray));
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
