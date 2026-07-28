#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 128;
constexpr int kCell = 8;
constexpr int kBins = 9;
using CellHistograms = std::vector<std::vector<std::vector<float>>>;

cv::Mat makeDemoImage() {
    cv::Mat image(kHeight, kWidth, CV_8UC3, cv::Scalar(28, 28, 28));
    cv::circle(image, cv::Point(kWidth / 2, 20), 9,
               cv::Scalar(225, 225, 225), -1);
    cv::ellipse(image, cv::Point(kWidth / 2, 54), cv::Size(13, 25),
                0, 0, 360, cv::Scalar(205, 205, 205), -1);
    cv::line(image, cv::Point(24, 44), cv::Point(8, 76),
             cv::Scalar(220, 220, 220), 7, cv::LINE_AA);
    cv::line(image, cv::Point(40, 44), cv::Point(56, 72),
             cv::Scalar(220, 220, 220), 7, cv::LINE_AA);
    cv::line(image, cv::Point(27, 74), cv::Point(20, 119),
             cv::Scalar(230, 230, 230), 9, cv::LINE_AA);
    cv::line(image, cv::Point(37, 74), cv::Point(47, 119),
             cv::Scalar(230, 230, 230), 9, cv::LINE_AA);
    cv::rectangle(image, cv::Rect(0, 120, kWidth, kHeight - 120),
                  cv::Scalar(75, 75, 75), -1);
    return image;
}

cv::Mat prepareImage(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(kWidth, kHeight), 0, 0, cv::INTER_AREA);
    return resized;
}

CellHistograms computeCellHistograms(const cv::Mat& image) {
    const cv::Mat prepared = prepareImage(image);
    cv::Mat gray;
    cv::cvtColor(prepared, gray, cv::COLOR_BGR2GRAY);
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);
    cv::Mat gradientX;
    cv::Mat gradientY;
    cv::Sobel(grayFloat, gradientX, CV_32F, 1, 0, 1);
    cv::Sobel(grayFloat, gradientY, CV_32F, 0, 1, 1);

    cv::Mat magnitude;
    cv::Mat angle;
    cv::cartToPolar(gradientX, gradientY, magnitude, angle, true);
    constexpr int cellsX = kWidth / kCell;
    constexpr int cellsY = kHeight / kCell;
    CellHistograms histograms(
        cellsY,
        std::vector<std::vector<float>>(
            cellsX, std::vector<float>(kBins, 0.0F)));
    constexpr float binWidth = 180.0F / kBins;

    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const float unsignedAngle =
                std::fmod(angle.at<float>(y, x), 180.0F);
            const float position = unsignedAngle / binWidth;
            const int lower = static_cast<int>(std::floor(position)) % kBins;
            const int upper = (lower + 1) % kBins;
            const float upperWeight = position - std::floor(position);
            const float pixelMagnitude = magnitude.at<float>(y, x);
            auto& histogram =
                histograms[static_cast<std::size_t>(y / kCell)]
                          [static_cast<std::size_t>(x / kCell)];
            histogram[static_cast<std::size_t>(lower)] +=
                pixelMagnitude * (1.0F - upperWeight);
            histogram[static_cast<std::size_t>(upper)] +=
                pixelMagnitude * upperWeight;
        }
    }
    return histograms;
}

std::vector<float> computeDescriptor(const cv::Mat& image) {
    const CellHistograms histograms = computeCellHistograms(image);
    std::vector<float> descriptor;
    descriptor.reserve(3780);
    constexpr float epsilon = 1e-5F;

    for (int blockY = 0; blockY < kHeight / kCell - 1; ++blockY) {
        for (int blockX = 0; blockX < kWidth / kCell - 1; ++blockX) {
            std::vector<float> block;
            block.reserve(4 * kBins);
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const auto& cell =
                        histograms[static_cast<std::size_t>(blockY + dy)]
                                  [static_cast<std::size_t>(blockX + dx)];
                    block.insert(block.end(), cell.begin(), cell.end());
                }
            }

            float squaredNorm = epsilon * epsilon;
            for (float value : block) {
                squaredNorm += value * value;
            }
            const float firstNorm = std::sqrt(squaredNorm);
            squaredNorm = epsilon * epsilon;
            for (float& value : block) {
                value = std::min(value / firstNorm, 0.2F);
                squaredNorm += value * value;
            }
            const float secondNorm = std::sqrt(squaredNorm);
            for (float value : block) {
                descriptor.push_back(value / secondNorm);
            }
        }
    }

    if (descriptor.size() != 3780) {
        throw std::runtime_error(
            "Expected a 3780-value descriptor, got " +
            std::to_string(descriptor.size()));
    }
    return descriptor;
}

cv::Mat visualize(const cv::Mat& image) {
    const cv::Mat prepared = prepareImage(image);
    const CellHistograms histograms = computeCellHistograms(prepared);
    float maximum = 0.0F;
    for (const auto& row : histograms) {
        for (const auto& cell : row) {
            for (float value : cell) {
                maximum = std::max(maximum, value);
            }
        }
    }

    constexpr int scale = 4;
    cv::Mat canvas;
    cv::resize(prepared, canvas, cv::Size(), scale, scale, cv::INTER_NEAREST);
    canvas.convertTo(canvas, -1, 0.45);
    if (maximum <= 0.0F) {
        return canvas;
    }

    constexpr int cellsX = kWidth / kCell;
    constexpr int cellsY = kHeight / kCell;
    const float radius = kCell * scale * 0.42F;
    constexpr float pi = 3.14159265358979323846F;
    for (int cellY = 0; cellY < cellsY; ++cellY) {
        for (int cellX = 0; cellX < cellsX; ++cellX) {
            const cv::Point center(
                static_cast<int>((cellX + 0.5F) * kCell * scale),
                static_cast<int>((cellY + 0.5F) * kCell * scale));
            for (int bin = 0; bin < kBins; ++bin) {
                const float strength =
                    histograms[static_cast<std::size_t>(cellY)]
                              [static_cast<std::size_t>(cellX)]
                              [static_cast<std::size_t>(bin)] /
                    maximum;
                if (strength < 0.03F) {
                    continue;
                }
                // Voting treats bin 0 as centered at 0 degrees, bin 1 at
                // 20 degrees, and so on. Draw those same centers.
                const float radians =
                    bin * 180.0F / kBins * pi / 180.0F;
                const float dx = std::cos(radians) * radius * strength;
                const float dy = std::sin(radians) * radius * strength;
                cv::line(
                    canvas,
                    cv::Point(
                        cvRound(center.x - dx), cvRound(center.y - dy)),
                    cv::Point(
                        cvRound(center.x + dx), cvRound(center.y + dy)),
                    cv::Scalar(50, 230, 255), 1, cv::LINE_AA);
            }
        }
    }
    return canvas;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string keys =
        "{help h usage ?||Show this help message}"
        "{input||Optional input image; omitted uses a generated demo}"
        "{output-dir|output|Output directory}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    try {
        const std::string inputPath = parser.get<std::string>("input");
        const std::string outputDir = parser.get<std::string>("output-dir");
        cv::Mat image = inputPath.empty()
            ? makeDemoImage()
            : cv::imread(inputPath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not read input image: " + inputPath);
        }
        image = prepareImage(image);
        const std::vector<float> descriptor = computeDescriptor(image);
        const cv::Mat visualization = visualize(image);

        cv::utils::fs::createDirectories(outputDir);
        const std::string inputOutput = outputDir + "/hog-input.png";
        const std::string visualizationOutput =
            outputDir + "/hog-visualization.png";
        const std::string descriptorOutput =
            outputDir + "/hog-descriptor.yml";
        if (!cv::imwrite(inputOutput, image) ||
            !cv::imwrite(visualizationOutput, visualization)) {
            throw std::runtime_error("Could not write output images.");
        }
        cv::FileStorage storage(descriptorOutput, cv::FileStorage::WRITE);
        storage << "descriptor" << descriptor;
        storage.release();

        double squaredNorm = 0.0;
        for (float value : descriptor) {
            squaredNorm += value * value;
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"window\":[64,128],\"blocks\":[7,15],"
                  << "\"cells_per_block\":4,\"bins\":9,"
                  << "\"descriptor_length\":" << descriptor.size()
                  << ",\"descriptor_l2_norm\":" << std::sqrt(squaredNorm)
                  << "}\n";
        return 0;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
    }
    return 2;
}
