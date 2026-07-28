#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>

namespace fs = std::filesystem;

cv::Mat preprocessImage(const cv::Mat& image, const std::string& mode) {
    if (image.empty()) {
        throw std::invalid_argument(
            "preprocessImage expects a non-empty image");
    }
    if (mode == "none") {
        return image.clone();
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    if (mode == "gray") {
        return gray;
    }
    if (mode == "otsu") {
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
        cv::Mat thresholded;
        cv::threshold(
            blurred, thresholded, 0, 255,
            cv::THRESH_BINARY | cv::THRESH_OTSU);
        return thresholded;
    }
    if (mode == "adaptive") {
        cv::Mat thresholded;
        cv::adaptiveThreshold(
            gray, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY, 31, 15);
        return thresholded;
    }
    throw std::invalid_argument(
        "preprocess must be one of: none, gray, otsu, adaptive");
}

std::string recognizeText(
    const cv::Mat& prepared, const std::string& language, int oem, int psm) {
    if (oem < 0 || oem > 3) {
        throw std::invalid_argument("oem must be in the range [0, 3]");
    }
    if (psm < 0 || psm > 13) {
        throw std::invalid_argument("psm must be in the range [0, 13]");
    }

    tesseract::TessBaseAPI ocr;
    if (ocr.Init(
            nullptr, language.c_str(),
            static_cast<tesseract::OcrEngineMode>(oem)) != 0) {
        throw std::runtime_error(
            "Tesseract could not initialize language: " + language);
    }
    ocr.SetPageSegMode(static_cast<tesseract::PageSegMode>(psm));

    cv::Mat tesseract_image;
    if (prepared.channels() == 3) {
        cv::cvtColor(prepared, tesseract_image, cv::COLOR_BGR2RGB);
    } else {
        tesseract_image = prepared;
    }
    ocr.SetImage(
        tesseract_image.data, tesseract_image.cols, tesseract_image.rows,
        tesseract_image.channels(),
        static_cast<int>(tesseract_image.step));

    std::unique_ptr<char[]> utf8_text(ocr.GetUTF8Text());
    if (!utf8_text) {
        throw std::runtime_error("Tesseract returned no text buffer");
    }
    std::string text(utf8_text.get());
    ocr.End();
    return text;
}

void printUsage(const char* program) {
    std::cerr << "Usage: " << program
              << " IMAGE [--preprocess none|gray|otsu|adaptive] "
                 "[--lang CODE] [--oem 0..3] [--psm 0..13] "
                 "[--output FILE] [--save-preprocessed FILE]\n";
}

bool parseInteger(const std::string& value, int& parsed) {
    try {
        std::size_t consumed = 0;
        parsed = std::stoi(value, &consumed);
        return consumed == value.size();
    } catch (const std::exception&) {
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 2;
    }

    fs::path image_path;
    fs::path output_path;
    fs::path prepared_path;
    std::string preprocessing = "gray";
    std::string language = "eng";
    int oem = 1;
    int psm = 6;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--preprocess" && index + 1 < argc) {
            preprocessing = argv[++index];
        } else if (argument == "--lang" && index + 1 < argc) {
            language = argv[++index];
        } else if (argument == "--oem" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (!parseInteger(value, oem)) {
                std::cerr << "error: invalid integer for --oem: "
                          << value << '\n';
                return 2;
            }
        } else if (argument == "--psm" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (!parseInteger(value, psm)) {
                std::cerr << "error: invalid integer for --psm: "
                          << value << '\n';
                return 2;
            }
        } else if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
        } else if (argument == "--save-preprocessed" && index + 1 < argc) {
            prepared_path = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (!argument.empty() && argument.front() != '-' &&
                   image_path.empty()) {
            image_path = argument;
        } else {
            printUsage(argv[0]);
            return 2;
        }
    }

    if (image_path.empty()) {
        printUsage(argv[0]);
        return 2;
    }

    try {
        const cv::Mat image =
            cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error(
                "Could not read input image: " + image_path.string());
        }
        const cv::Mat prepared = preprocessImage(image, preprocessing);
        const std::string text =
            recognizeText(prepared, language, oem, psm);

        if (!prepared_path.empty()) {
            if (prepared_path.has_parent_path()) {
                fs::create_directories(prepared_path.parent_path());
            }
            if (!cv::imwrite(prepared_path.string(), prepared)) {
                throw std::runtime_error(
                    "OpenCV could not write preprocessed image: " +
                    prepared_path.string());
            }
        }
        if (!output_path.empty()) {
            if (output_path.has_parent_path()) {
                fs::create_directories(output_path.parent_path());
            }
            std::ofstream output(output_path);
            if (!output) {
                throw std::runtime_error(
                    "Could not open text output: " + output_path.string());
            }
            output << text;
        }
        std::cout << text;
        if (!text.empty() && text.back() != '\n') {
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
