#include "face_detection.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

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
    std::filesystem::path output_dir =
        std::filesystem::path(FACE_SOURCE_DIR) / "output";
    std::string mode = "auto";
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
        } else if (argument == "--output-dir") {
            options.output_dir = requireValue(index, argc, argv);
        } else if (argument == "--mode") {
            options.mode = requireValue(index, argc, argv);
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
        << "  --output-dir PATH        Output directory\n"
        << "  --device cpu|cuda        DNN execution device\n"
        << "  --score-threshold VALUE  Face score threshold\n"
        << "  --nms-threshold VALUE    NMS IoU threshold\n"
        << "  --top-k N                Boxes retained before NMS\n"
        << "  --max-frames N           Video only; zero means complete input\n"
        << "  --display                Open an interactive window\n"
        << "  --no-display             Run headlessly (default)\n"
        << "  --validate               Check stable output invariants\n";
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
            "Saved video has unexpected dimensions or frame count.");
    }
}

void runImage(
    const Options& options,
    learnopencv::face::YuNetDetector& detector) {
    const cv::Mat frame =
        cv::imread(options.input.string(), cv::IMREAD_COLOR);
    if (frame.empty()) {
        throw std::runtime_error(
            "Could not read input image: " + options.input.string());
    }
    const std::vector<learnopencv::face::Detection> detections =
        detector.detect(frame);
    if (options.validate) {
        learnopencv::face::validate(frame, detections);
    }
    const cv::Mat output =
        learnopencv::face::draw(frame, detections, detector.name());

    std::filesystem::create_directories(options.output_dir);
    const std::filesystem::path output_path =
        options.output_dir / "yunet-image.jpg";
    if (!cv::imwrite(output_path.string(), output)) {
        throw std::runtime_error(
            "Could not write output image: " + output_path.string());
    }

    if (options.validate) {
        const cv::Mat saved =
            cv::imread(output_path.string(), cv::IMREAD_COLOR);
        if (saved.empty() || saved.size() != frame.size()) {
            throw std::runtime_error(
                "Saved image is unreadable or changed dimensions.");
        }
        std::cout
            << "VALIDATION PASSED: mode=image faces="
            << detections.size() << '\n';
    }
    if (options.display) {
        cv::imshow("YuNet Face Detection", output);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    std::cout
        << "FACE RESULT: detector=YuNet faces="
        << detections.size() << '\n';
    std::cout << "Saved output: " << output_path << '\n';
}

void runVideo(
    const Options& options,
    learnopencv::face::YuNetDetector& detector) {
    cv::VideoCapture capture(options.input.string());
    if (!capture.isOpened()) {
        throw std::runtime_error(
            "Could not open input video: " + options.input.string());
    }
    const cv::Size frame_size{
        cvRound(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
    };
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        throw std::runtime_error("Input video has invalid dimensions.");
    }
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 25.0;
    }

    std::filesystem::create_directories(options.output_dir);
    const std::filesystem::path output_path =
        options.output_dir / "yunet-video.avi";
    cv::VideoWriter writer(
        output_path.string(),
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        frame_size);
    if (!writer.isOpened()) {
        throw std::runtime_error(
            "Could not create output video: " + output_path.string());
    }

    int processed = 0;
    long long total_faces = 0;
    cv::Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            throw std::runtime_error(
                "Decoded an empty frame at index " +
                std::to_string(processed) + ".");
        }
        if (frame.size() != frame_size) {
            throw std::runtime_error(
                "A decoded frame changed dimensions.");
        }
        const std::vector<learnopencv::face::Detection> detections =
            detector.detect(frame);
        if (options.validate) {
            learnopencv::face::validate(frame, detections);
        }
        const cv::Mat output =
            learnopencv::face::draw(frame, detections, detector.name());
        writer.write(output);
        ++processed;
        total_faces += static_cast<long long>(detections.size());

        if (options.display) {
            cv::imshow("YuNet Face Detection", output);
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
        validateWrittenVideo(output_path, frame_size, processed);
        std::cout
            << "VALIDATION PASSED: mode=video frames=" << processed
            << " size=" << frame_size.width
            << 'x' << frame_size.height << '\n';
    }
    std::cout
        << "FACE VIDEO RESULT: detector=YuNet frames=" << processed
        << " total_faces=" << total_faces << '\n';
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
        learnopencv::face::YuNetDetector detector(
            options.model,
            options.score_threshold,
            options.nms_threshold,
            options.top_k,
            options.device);
        const std::string mode = inferMode(options.input, options.mode);
        if (mode == "image") {
            runImage(options, detector);
        } else {
            runVideo(options, detector);
        }
        std::cout << "OpenCV version: " << CV_VERSION << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
