#include "pose_estimation.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef OPENPOSE_SOURCE_DIR
#define OPENPOSE_SOURCE_DIR "."
#endif

namespace {

struct Options {
    std::filesystem::path input =
        std::filesystem::path(OPENPOSE_SOURCE_DIR) / "sample_video.mp4";
    std::filesystem::path model =
        std::filesystem::path(OPENPOSE_SOURCE_DIR) /
        "models" /
        "pose_estimation_mediapipe_2023mar.onnx";
    std::filesystem::path output_dir =
        std::filesystem::path(OPENPOSE_SOURCE_DIR) / "output";
    std::string device = "cpu";
    float score_threshold = 0.5F;
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
        } else if (argument == "--device") {
            options.device = requireValue(index, argc, argv);
        } else if (argument == "--score-threshold") {
            options.score_threshold =
                std::stof(requireValue(index, argc, argv));
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
    if (options.score_threshold < 0.0F || options.score_threshold > 1.0F) {
        throw std::invalid_argument(
            "--score-threshold must be between 0 and 1.");
    }
    if (options.max_frames < 0) {
        throw std::invalid_argument("--max-frames cannot be negative.");
    }
    return options;
}

void printHelp(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --input PATH             Input video\n"
        << "  --model PATH             MediaPipe Pose ONNX model\n"
        << "  --output-dir PATH        Directory for pose-video.avi\n"
        << "  --device cpu|cuda        DNN execution device\n"
        << "  --score-threshold VALUE  Landmark probability threshold\n"
        << "  --max-frames N           Zero processes the complete video\n"
        << "  --display                Open an interactive window\n"
        << "  --no-display             Run headlessly (default)\n"
        << "  --validate               Check stable output invariants\n";
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
    if (actual_size != expected_size) {
        throw std::runtime_error("Output video dimensions changed.");
    }
    if (actual_frames != expected_frames) {
        throw std::runtime_error(
            "Output video frame count does not match processed frames.");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printHelp(argv[0]);
            return 0;
        }

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
            options.output_dir / "pose-video.avi";
        cv::VideoWriter writer(
            output_path.string(),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            frame_size);
        if (!writer.isOpened()) {
            throw std::runtime_error(
                "Could not create output video: " + output_path.string());
        }

        learnopencv::pose::PoseEstimator estimator(
            options.model, options.device);
        int processed = 0;
        long long total_visible = 0;
        long long total_edges = 0;
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

            const learnopencv::pose::PoseResult result =
                estimator.infer(frame);
            cv::Mat output = frame.clone();
            const learnopencv::pose::DrawMetrics metrics =
                learnopencv::pose::draw(
                    output, result, options.score_threshold);
            if (options.validate) {
                learnopencv::pose::validate(frame, result, metrics);
            }
            writer.write(output);
            ++processed;
            total_visible += metrics.visible_count;
            total_edges += metrics.edge_count;

            if (options.display) {
                cv::imshow("MediaPipe Pose", output);
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
                << "VALIDATION PASSED: frames=" << processed
                << " size=" << frame_size.width
                << 'x' << frame_size.height << '\n';
        }

        std::cout << "OpenCV version: " << CV_VERSION << '\n';
        std::cout
            << "POSE VIDEO RESULT: frames=" << processed
            << " total_visible=" << total_visible
            << " total_edges=" << total_edges << '\n';
        std::cout << "Saved output: " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
