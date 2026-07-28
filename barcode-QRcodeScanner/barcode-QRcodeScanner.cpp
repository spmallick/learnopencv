#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/barcode.hpp>

namespace fs = std::filesystem;

struct CodeResult {
    std::string type;
    std::string data;
    std::vector<cv::Point2f> points;
};

std::vector<std::vector<cv::Point2f>> polygonsFromMat(const cv::Mat& points) {
    std::vector<std::vector<cv::Point2f>> polygons;
    if (points.empty()) {
        return polygons;
    }
    if (points.type() != CV_32FC2 || points.cols != 4) {
        throw std::runtime_error("OpenCV returned an unexpected polygon layout");
    }

    polygons.reserve(points.rows);
    for (int row = 0; row < points.rows; ++row) {
        const cv::Point2f* begin = points.ptr<cv::Point2f>(row);
        polygons.emplace_back(begin, begin + points.cols);
    }
    return polygons;
}

std::vector<CodeResult> decodeQrCodes(const cv::Mat& image) {
    cv::QRCodeDetector detector;
    std::vector<std::string> decoded_info;
    cv::Mat points;
    std::vector<CodeResult> results;

    if (detector.detectAndDecodeMulti(image, decoded_info, points)) {
        const auto polygons = polygonsFromMat(points);
        const std::size_t count = std::min(decoded_info.size(), polygons.size());
        for (std::size_t index = 0; index < count; ++index) {
            if (!decoded_info[index].empty()) {
                results.push_back(
                    {"QR_CODE", decoded_info[index], polygons[index]});
            }
        }
    }
    if (!results.empty()) {
        return results;
    }

    cv::Mat single_points;
    const std::string data = detector.detectAndDecode(image, single_points);
    const auto polygons = polygonsFromMat(single_points);
    if (!data.empty() && !polygons.empty()) {
        results.push_back({"QR_CODE", data, polygons.front()});
    }
    return results;
}

std::vector<CodeResult> decodeBarcodes(const cv::Mat& image) {
    cv::barcode::BarcodeDetector detector;
    std::vector<std::string> decoded_info;
    std::vector<std::string> decoded_types;
    cv::Mat points;
    std::vector<CodeResult> results;

    if (!detector.detectAndDecodeWithType(
            image, decoded_info, decoded_types, points)) {
        return results;
    }

    const auto polygons = polygonsFromMat(points);
    const std::size_t count =
        std::min({decoded_info.size(), decoded_types.size(), polygons.size()});
    for (std::size_t index = 0; index < count; ++index) {
        if (!decoded_info[index].empty()) {
            results.push_back(
                {decoded_types[index], decoded_info[index], polygons[index]});
        }
    }
    return results;
}

cv::Mat annotateCodes(
    const cv::Mat& image, const std::vector<CodeResult>& results) {
    cv::Mat annotated = image.clone();
    for (const CodeResult& result : results) {
        std::vector<cv::Point> polygon;
        polygon.reserve(result.points.size());
        for (const cv::Point2f& point : result.points) {
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
        }
        cv::polylines(annotated, polygon, true, cv::Scalar(30, 190, 30), 3,
                      cv::LINE_AA);

        const auto leftmost = std::min_element(
            polygon.begin(), polygon.end(),
            [](const cv::Point& left, const cv::Point& right) {
                return left.x < right.x;
            });
        const auto topmost = std::min_element(
            polygon.begin(), polygon.end(),
            [](const cv::Point& left, const cv::Point& right) {
                return left.y < right.y;
            });
        const cv::Point anchor(
            leftmost == polygon.end() ? 0 : leftmost->x,
            std::max(24, (topmost == polygon.end() ? 24 : topmost->y - 10)));
        cv::putText(
            annotated, result.type + ": " + result.data, anchor,
            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(25, 25, 220), 2,
            cv::LINE_AA);
    }
    return annotated;
}

void printUsage(const char* program) {
    std::cerr << "Usage: " << program
              << " [image] [--output FILE] [--no-qr] [--no-barcode] "
                 "[--display]\n";
}

int main(int argc, char** argv) {
    fs::path image_path = "zbar-test.jpg";
    fs::path output_path = "output/decoded-codes.png";
    bool scan_qr = true;
    bool scan_barcodes = true;
    bool display = false;
    bool positional_image_seen = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
        } else if (argument == "--no-qr") {
            scan_qr = false;
        } else if (argument == "--no-barcode") {
            scan_barcodes = false;
        } else if (argument == "--display") {
            display = true;
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (!argument.empty() && argument.front() != '-' &&
                   !positional_image_seen) {
            image_path = argument;
            positional_image_seen = true;
        } else {
            printUsage(argv[0]);
            return 2;
        }
    }

    if (!scan_qr && !scan_barcodes) {
        std::cerr << "error: both decoders are disabled\n";
        return 2;
    }

    try {
        const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not read input image: " +
                                     image_path.string());
        }

        std::vector<CodeResult> results;
        if (scan_qr) {
            auto qr_results = decodeQrCodes(image);
            results.insert(
                results.end(), qr_results.begin(), qr_results.end());
        }
        if (scan_barcodes) {
            auto barcode_results = decodeBarcodes(image);
            results.insert(
                results.end(), barcode_results.begin(), barcode_results.end());
        }

        for (const CodeResult& result : results) {
            std::cout << result.type << '\t' << result.data << '\n';
        }
        std::cout << "decoded_count=" << results.size() << '\n';

        const cv::Mat annotated = annotateCodes(image, results);
        if (output_path.has_parent_path()) {
            fs::create_directories(output_path.parent_path());
        }
        if (!cv::imwrite(output_path.string(), annotated)) {
            throw std::runtime_error(
                "OpenCV could not write output image: " +
                output_path.string());
        }
        std::cout << "output=" << fs::absolute(output_path) << '\n';

        if (display) {
            cv::imshow("Decoded codes", annotated);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
