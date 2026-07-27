#include "face_detection.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FACE_SOURCE_DIR
#define FACE_SOURCE_DIR "."
#endif

namespace {

struct Options {
    std::filesystem::path input =
        std::filesystem::path(FACE_SOURCE_DIR) / "videos" / "baby.mp4";
    std::filesystem::path model =
        std::filesystem::path(FACE_SOURCE_DIR) /
        "models" /
        "face_detection_yunet_2026may.onnx";
    std::filesystem::path cascade =
        std::filesystem::path(FACE_SOURCE_DIR) /
        "models" /
        "haarcascade_frontalface_default.xml";
    std::filesystem::path output_dir =
        std::filesystem::path(FACE_SOURCE_DIR) / "output-comparison";
    std::string mode = "auto";
    std::string detectors = "yunet";
    std::string device = "cpu";
    float score_threshold = 0.7F;
    float nms_threshold = 0.3F;
    int top_k = 5000;
    int max_frames = 0;
    bool display = false;
    bool validate = false;
    bool help = false;
};

std::string requireValue(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            "Missing value after " + std::string(argv[index]));
    }
    ++index;
    return argv[index];
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--input") {
            options.input = requireValue(index, argc, argv);
        } else if (argument == "--model") {
            options.model = requireValue(index, argc, argv);
        } else if (argument == "--cascade") {
            options.cascade = requireValue(index, argc, argv);
        } else if (argument == "--output-dir") {
            options.output_dir = requireValue(index, argc, argv);
        } else if (argument == "--mode") {
            options.mode = requireValue(index, argc, argv);
        } else if (argument == "--detectors") {
            options.detectors = requireValue(index, argc, argv);
        } else if (argument == "--device") {
            options.device = requireValue(index, argc, argv);
        } else if (argument == "--score-threshold") {
            options.score_threshold =
                std::stof(requireValue(index, argc, argv));
        } else if (argument == "--nms-threshold") {
            options.nms_threshold =
                std::stof(requireValue(index, argc, argv));
        } else if (argument == "--top-k") {
            options.top_k =
                std::stoi(requireValue(index, argc, argv));
        } else if (argument == "--max-frames") {
            options.max_frames =
                std::stoi(requireValue(index, argc, argv));
        } else if (argument == "--display") {
            options.display = true;
        } else if (argument == "--no-display") {
            options.display = false;
        } else if (argument == "--validate") {
            options.validate = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    if (options.mode != "auto" &&
        options.mode != "image" &&
        options.mode != "video") {
        throw std::invalid_argument(
            "--mode must be auto, image, or video.");
    }
    if (options.max_frames < 0) {
        throw std::invalid_argument("--max-frames cannot be negative.");
    }
    return options;
}

void printHelp(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --input PATH             Input image or video\n"
        << "  --mode auto|image|video  Input type (default: auto)\n"
        << "  --model PATH             Dynamic YuNet ONNX model\n"
        << "  --cascade PATH           Optional Haar XML\n"
        << "  --detectors LIST         Comma-separated yunet,haar,hog\n"
        << "  --output-dir PATH        Comparison output directory\n"
        << "  --device cpu|cuda        YuNet DNN device\n"
        << "  --score-threshold VALUE  YuNet score threshold\n"
        << "  --nms-threshold VALUE    YuNet NMS threshold\n"
        << "  --top-k N                YuNet boxes before NMS\n"
        << "  --max-frames N           Video only; zero means complete\n"
        << "  --display                Open an interactive window\n"
        << "  --no-display             Run headlessly (default)\n"
        << "  --validate               Check stable output invariants\n";
}

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(
        value.begin(),
        std::find_if(value.begin(), value.end(), not_space));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), not_space).base(),
        value.end());
    return value;
}

std::string lower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

std::vector<std::string> splitDetectorNames(const std::string& list) {
    std::vector<std::string> names;
    std::set<std::string> seen;
    std::istringstream stream(list);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::string name = lower(trim(item));
        if (name.empty()) {
            continue;
        }
        if (!seen.insert(name).second) {
            throw std::invalid_argument(
                "--detectors cannot contain duplicates.");
        }
        names.push_back(name);
    }
    if (names.empty()) {
        throw std::invalid_argument(
            "--detectors must name at least one detector.");
    }
    return names;
}

std::vector<std::unique_ptr<learnopencv::face::Detector>> buildDetectors(
    const Options& options) {
    std::vector<std::unique_ptr<learnopencv::face::Detector>> detectors;
    for (const std::string& name : splitDetectorNames(options.detectors)) {
        if (name == "yunet") {
            detectors.push_back(
                std::make_unique<learnopencv::face::YuNetDetector>(
                    options.model,
                    options.score_threshold,
                    options.nms_threshold,
                    options.top_k,
                    options.device));
            continue;
        }
        if (name == "haar") {
#if CV_VERSION_MAJOR < 5
            detectors.push_back(
                std::make_unique<learnopencv::face::HaarDetector>(
                    options.cascade));
            continue;
#else
            throw std::runtime_error(
                "Haar is optional and unavailable in this OpenCV 5 build.");
#endif
        }
        if (name == "hog" || name == "dlib" || name == "dlib-hog") {
#ifdef FACE_WITH_DLIB
            detectors.push_back(
                std::make_unique<learnopencv::face::DlibHogDetector>());
            continue;
#else
            throw std::runtime_error(
                "dlib was not found when this target was configured.");
#endif
        }
        throw std::invalid_argument(
            "Unknown detector '" + name +
            "'. Choose yunet, haar, or hog.");
    }
    return detectors;
}

std::string inferMode(
    const std::filesystem::path& input,
    const std::string& requested) {
    if (requested != "auto") {
        return requested;
    }
    const std::string extension = lower(input.extension().string());
    if (extension == ".bmp" ||
        extension == ".jpeg" ||
        extension == ".jpg" ||
        extension == ".png" ||
        extension == ".tif" ||
        extension == ".tiff" ||
        extension == ".webp") {
        return "image";
    }
    return "video";
}

std::pair<cv::Mat, std::vector<int>> compareFrame(
    const cv::Mat& frame,
    std::vector<std::unique_ptr<learnopencv::face::Detector>>& detectors,
    bool should_validate) {
    std::vector<cv::Mat> panels;
    std::vector<int> counts;
    panels.reserve(detectors.size());
    counts.reserve(detectors.size());
    for (const auto& detector : detectors) {
        const std::vector<learnopencv::face::Detection> detections =
            detector->detect(frame);
        if (should_validate) {
            learnopencv::face::validate(frame, detections);
        }
        panels.push_back(
            learnopencv::face::draw(frame, detections, detector->name()));
        counts.push_back(static_cast<int>(detections.size()));
    }
    cv::Mat comparison = learnopencv::face::joinPanels(panels);
    const cv::Size expected_size{
        frame.cols * static_cast<int>(detectors.size()),
        frame.rows,
    };
    if (comparison.size() != expected_size) {
        throw std::runtime_error(
            "Comparison panel dimensions are incorrect.");
    }
    return {comparison, counts};
}

void printCounts(
    const std::string& prefix,
    const std::vector<std::unique_ptr<learnopencv::face::Detector>>& detectors,
    const std::vector<long long>& counts) {
    std::cout << prefix;
    for (std::size_t index = 0; index < detectors.size(); ++index) {
        std::cout
            << ' ' << detectors[index]->name()
            << '=' << counts[index];
    }
    std::cout << '\n';
}

void validateWrittenVideo(
    const std::filesystem::path& output_path,
    const cv::Size& expected_size,
    int expected_frames) {
    cv::VideoCapture capture(output_path.string());
    if (!capture.isOpened()) {
        throw std::runtime_error(
            "Could not reopen output video: " + output_path.string());
    }
    const cv::Size actual_size{
        cvRound(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
    };
    const int actual_frames =
        cvRound(capture.get(cv::CAP_PROP_FRAME_COUNT));
    capture.release();
    if (actual_size != expected_size || actual_frames != expected_frames) {
        throw std::runtime_error(
            "Saved comparison video has unexpected geometry or frame count.");
    }
}

void runImage(
    const Options& options,
    std::vector<std::unique_ptr<learnopencv::face::Detector>>& detectors) {
    const cv::Mat frame =
        cv::imread(options.input.string(), cv::IMREAD_COLOR);
    if (frame.empty()) {
        throw std::runtime_error(
            "Could not read input image: " + options.input.string());
    }
    auto [comparison, counts] =
        compareFrame(frame, detectors, options.validate);

    std::filesystem::create_directories(options.output_dir);
    const std::filesystem::path output_path =
        options.output_dir / "comparison-image.jpg";
    if (!cv::imwrite(output_path.string(), comparison)) {
        throw std::runtime_error(
            "Could not write output image: " + output_path.string());
    }
    if (options.validate) {
        const cv::Mat saved =
            cv::imread(output_path.string(), cv::IMREAD_COLOR);
        if (saved.empty() || saved.size() != comparison.size()) {
            throw std::runtime_error(
                "Saved comparison image is unreadable or resized.");
        }
        std::cout
            << "VALIDATION PASSED: mode=image panels="
            << detectors.size()
            << " size=" << comparison.cols
            << 'x' << comparison.rows << '\n';
    }
    if (options.display) {
        cv::imshow("Face Detection Comparison", comparison);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::vector<long long> totals(
        counts.begin(), counts.end());
    printCounts("COMPARISON RESULT:", detectors, totals);
    std::cout << "Saved output: " << output_path << '\n';
}

void runVideo(
    const Options& options,
    std::vector<std::unique_ptr<learnopencv::face::Detector>>& detectors) {
    cv::VideoCapture capture(options.input.string());
    if (!capture.isOpened()) {
        throw std::runtime_error(
            "Could not open input video: " + options.input.string());
    }
    const cv::Size input_size{
        cvRound(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
    };
    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::runtime_error("Input video has invalid dimensions.");
    }
    const cv::Size output_size{
        input_size.width * static_cast<int>(detectors.size()),
        input_size.height,
    };
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 25.0;
    }

    std::filesystem::create_directories(options.output_dir);
    const std::filesystem::path output_path =
        options.output_dir / "comparison-video.avi";
    cv::VideoWriter writer(
        output_path.string(),
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        output_size);
    if (!writer.isOpened()) {
        throw std::runtime_error(
            "Could not create output video: " + output_path.string());
    }

    int processed = 0;
    std::vector<long long> totals(detectors.size(), 0);
    cv::Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            throw std::runtime_error(
                "Decoded an empty frame at index " +
                std::to_string(processed) + ".");
        }
        auto [comparison, counts] =
            compareFrame(frame, detectors, options.validate);
        writer.write(comparison);
        ++processed;
        for (std::size_t index = 0; index < totals.size(); ++index) {
            totals[index] += counts[index];
        }

        if (options.display) {
            cv::imshow("Face Detection Comparison", comparison);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        if (options.max_frames > 0 &&
            processed >= options.max_frames) {
            break;
        }
    }
    capture.release();
    writer.release();
    if (options.display) {
        cv::destroyAllWindows();
    }
    if (processed == 0) {
        throw std::runtime_error("No frames were decoded.");
    }
    if (options.validate) {
        validateWrittenVideo(output_path, output_size, processed);
        std::cout
            << "VALIDATION PASSED: mode=video panels="
            << detectors.size()
            << " frames=" << processed
            << " size=" << output_size.width
            << 'x' << output_size.height << '\n';
    }
    printCounts(
        "COMPARISON VIDEO RESULT: frames=" +
            std::to_string(processed),
        detectors,
        totals);
    std::cout << "Saved output: " << output_path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printHelp(argv[0]);
            return 0;
        }
        auto detectors = buildDetectors(options);
        const std::string mode = inferMode(options.input, options.mode);
        if (mode == "image") {
            runImage(options, detectors);
        } else {
            runVideo(options, detectors);
        }
        std::cout << "OpenCV version: " << CV_VERSION << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
