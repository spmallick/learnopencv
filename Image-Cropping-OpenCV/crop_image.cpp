#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

cv::Rect validateRoi(const cv::Mat& image, int x, int y, int width, int height) {
    if (image.empty()) throw std::invalid_argument("image must be non-empty");
    if (x < 0 || y < 0 || width <= 0 || height <= 0) {
        throw std::invalid_argument("x/y must be non-negative and width/height must be positive");
    }
    if (x >= image.cols || y >= image.rows ||
        width > image.cols - x || height > image.rows - y) {
        throw std::invalid_argument("ROI exceeds image bounds");
    }
    return cv::Rect(x, y, width, height);
}

cv::Mat cropImage(const cv::Mat& image, int x, int y, int width, int height) {
    return image(validateRoi(image, x, y, width, height)).clone();
}

struct Tile {
    int row;
    int column;
    cv::Mat image;
};

std::vector<Tile> extractTiles(const cv::Mat& image, int tileWidth, int tileHeight) {
    if (tileWidth <= 0 || tileHeight <= 0) {
        throw std::invalid_argument("tile dimensions must be positive");
    }
    std::vector<Tile> tiles;
    for (int y = 0, row = 0; y < image.rows; y += tileHeight, ++row) {
        for (int x = 0, column = 0; x < image.cols; x += tileWidth, ++column) {
            const int width = std::min(tileWidth, image.cols - x);
            const int height = std::min(tileHeight, image.rows - y);
            tiles.push_back({row, column, image(cv::Rect(x, y, width, height)).clone()});
        }
    }
    return tiles;
}

cv::Mat makeCropComparison(const cv::Mat& image, const cv::Mat& cropped,
                           const cv::Rect& roi) {
    cv::Mat source = image.clone();
    cv::rectangle(source, roi, cv::Scalar(0, 255, 255), 4);
    cv::Mat cropPanel = cv::Mat::zeros(image.size(), image.type());
    const double scale = std::min(
        static_cast<double>(image.cols) / cropped.cols,
        static_cast<double>(image.rows) / cropped.rows);
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(), scale, scale, cv::INTER_NEAREST);
    const int x = (cropPanel.cols - resized.cols) / 2;
    const int y = (cropPanel.rows - resized.rows) / 2;
    resized.copyTo(cropPanel(cv::Rect(x, y, resized.cols, resized.rows)));
    std::vector<cv::Mat> panels{source, cropPanel};
    const std::vector<std::string> labels{"Validated ROI", "Cropped pixels"};
    for (std::size_t i = 0; i < panels.size(); ++i) {
        cv::rectangle(panels[i], cv::Rect(0, 0, panels[i].cols, 42), cv::Scalar::all(0), -1);
        cv::putText(panels[i], labels[i], cv::Point(14, 29), cv::FONT_HERSHEY_SIMPLEX,
                    0.72, cv::Scalar::all(255), 2, cv::LINE_AA);
    }
    cv::Mat comparison;
    cv::hconcat(panels, comparison);
    return comparison;
}

cv::Mat makeTileContactSheet(const std::vector<Tile>& tiles, int tileWidth, int tileHeight) {
    if (tiles.empty()) throw std::invalid_argument("at least one tile is required");
    int rows = 0;
    int columns = 0;
    for (const Tile& tile : tiles) {
        rows = std::max(rows, tile.row + 1);
        columns = std::max(columns, tile.column + 1);
    }
    cv::Mat sheet = cv::Mat::zeros(rows * tileHeight, columns * tileWidth, CV_8UC3);
    for (const Tile& tile : tiles) {
        const cv::Rect target(tile.column * tileWidth, tile.row * tileHeight,
                              tile.image.cols, tile.image.rows);
        tile.image.copyTo(sheet(target));
        cv::rectangle(sheet,
                      cv::Rect(tile.column * tileWidth, tile.row * tileHeight,
                               tileWidth, tileHeight),
                      cv::Scalar::all(255), 1);
    }
    return sheet;
}

void requireWrite(const fs::path& path, const cv::Mat& image) {
    fs::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("could not write " + path.string());
    }
}

int selfTest() {
    cv::Mat fixture(12, 16, CV_8UC3);
    for (int row = 0; row < fixture.rows; ++row) {
        for (int column = 0; column < fixture.cols; ++column) {
            fixture.at<cv::Vec3b>(row, column) =
                cv::Vec3b(static_cast<unsigned char>(column * 11),
                          static_cast<unsigned char>(row * 17),
                          static_cast<unsigned char>((row + column) * 7));
        }
    }
    const cv::Mat cropped = cropImage(fixture, 3, 2, 7, 5);
    cv::Mat difference;
    cv::absdiff(cropped, fixture(cv::Rect(3, 2, 7, 5)), difference);
    const auto tiles = extractTiles(fixture, 6, 5);
    bool rejectedOversizedWidth = false;
    try {
        static_cast<void>(
            validateRoi(fixture, 15, 0, std::numeric_limits<int>::max(), 5));
    } catch (const std::invalid_argument&) {
        rejectedOversizedWidth = true;
    }
    if (cropped.size() != cv::Size(7, 5) ||
        cv::countNonZero(difference.reshape(1)) != 0 || tiles.size() != 9 ||
        !rejectedOversizedWidth) {
        std::cerr << "self-test failed\n";
        return 1;
    }
    std::cout << "self-test passed: tiles=" << tiles.size() << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        fs::path input = fs::path(__FILE__).parent_path() / "assets" / "sample-scene.png";
        fs::path outputDir = fs::path(__FILE__).parent_path() / "outputs";
        int x = 160, y = 90, width = 320, height = 240;
        int tileWidth = 160, tileHeight = 140;
        bool display = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--self-test") return selfTest();
            if (arg == "--display") display = true;
            else if (arg == "--input" && i + 1 < argc) input = argv[++i];
            else if (arg == "--output-dir" && i + 1 < argc) outputDir = argv[++i];
            else if (arg == "--roi" && i + 4 < argc) {
                x = std::stoi(argv[++i]); y = std::stoi(argv[++i]);
                width = std::stoi(argv[++i]); height = std::stoi(argv[++i]);
            } else if (arg == "--tile-size" && i + 2 < argc) {
                tileWidth = std::stoi(argv[++i]); tileHeight = std::stoi(argv[++i]);
            } else if (arg != "--display") {
                throw std::invalid_argument("unknown or incomplete option: " + arg);
            }
        }
        const cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("could not read " + input.string());
        const cv::Rect roi = validateRoi(image, x, y, width, height);
        const cv::Mat cropped = cropImage(image, x, y, width, height);
        const auto tiles = extractTiles(image, tileWidth, tileHeight);
        requireWrite(outputDir / "cropped.png", cropped);
        const cv::Mat comparison = makeCropComparison(image, cropped, roi);
        requireWrite(outputDir / "crop-comparison.png", comparison);
        requireWrite(outputDir / "tile-contact-sheet.png",
                     makeTileContactSheet(tiles, tileWidth, tileHeight));
        for (const Tile& tile : tiles) {
            requireWrite(outputDir / "patches" /
                         ("patch-r" + std::to_string(tile.row) + "-c" +
                          std::to_string(tile.column) + ".png"), tile.image);
        }
        std::cout << "input=" << image.cols << "x" << image.rows
                  << " crop=" << cropped.cols << "x" << cropped.rows
                  << " tiles=" << tiles.size() << '\n';
        if (display) {
            cv::imshow("Crop and ROI", comparison);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
