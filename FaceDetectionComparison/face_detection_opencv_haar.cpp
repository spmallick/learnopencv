#include "face_detection.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

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
    std::filesystem::path cascade =
        std::filesystem::path(FACE_SOURCE_DIR) /
        "models" /
        "haarcascade_frontalface_default.xml";
    std::filesystem::path output_dir =
        std::filesystem::path(FACE_SOURCE_DIR) / "output-historical";
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
        } else if (argument == "--cascade") {
            options.cascade = requireValue(index, argc, argv);
        } else if (argument == "--output-dir") {
            options.output_dir = requireValue(index, argc, argv);
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
    if (options.max_frames < 0) {
        throw std::invalid_argument("--max-frames cannot be negative.");
    }
    return options;
}

void printHelp(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "Historical Haar video baseline (OpenCV 4 builds only).\n"
        << "  --input PATH       Input video\n"
        << "  --cascade PATH     Haar cascade XML\n"
        << "  --output-dir PATH  Output directory\n"
        << "  --max-frames N     Zero processes the complete video\n"
        << "  --display          Open an interactive window\n"
        << "  --no-display       Run headlessly (default)\n"
        << "  --validate         Check output invariants\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printHelp(argv[0]);
            return 0;
        }

        learnopencv::face::HaarDetector detector(options.cascade);
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
            options.output_dir / "haar-video.avi";
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
            const std::vector<learnopencv::face::Detection> detections =
                detector.detect(frame);
            if (options.validate) {
                learnopencv::face::validate(frame, detections);
            }
            const cv::Mat output =
                learnopencv::face::draw(
                    frame, detections, detector.name());
            writer.write(output);
            ++processed;
            total_faces += static_cast<long long>(detections.size());

            if (options.display) {
                cv::imshow("Historical Haar Face Detection", output);
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
            cv::VideoCapture check(output_path.string());
            const cv::Size actual_size{
                cvRound(check.get(cv::CAP_PROP_FRAME_WIDTH)),
                cvRound(check.get(cv::CAP_PROP_FRAME_HEIGHT)),
            };
            const int actual_frames =
                cvRound(check.get(cv::CAP_PROP_FRAME_COUNT));
            check.release();
            if (actual_size != frame_size || actual_frames != processed) {
                throw std::runtime_error(
                    "Saved Haar video failed output validation.");
            }
            std::cout
                << "VALIDATION PASSED: detector=Haar frames="
                << processed
                << " size=" << frame_size.width
                << 'x' << frame_size.height << '\n';
        }
        std::cout
            << "HAAR VIDEO RESULT: frames=" << processed
            << " total_faces=" << total_faces << '\n';
        std::cout << "Saved output: " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
