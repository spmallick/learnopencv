#include "pose_estimation.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

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
        std::filesystem::path(OPENPOSE_SOURCE_DIR) / "single.jpeg";
    std::filesystem::path model =
        std::filesystem::path(OPENPOSE_SOURCE_DIR) /
        "models" /
        "pose_estimation_mediapipe_2023mar.onnx";
    std::filesystem::path output_dir =
        std::filesystem::path(OPENPOSE_SOURCE_DIR) / "output";
    std::string device = "cpu";
    float score_threshold = 0.5F;
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
    return options;
}

void printHelp(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --input PATH             Input image\n"
        << "  --model PATH             MediaPipe Pose ONNX model\n"
        << "  --output-dir PATH        Directory for pose-image.jpg\n"
        << "  --device cpu|cuda        DNN execution device\n"
        << "  --score-threshold VALUE  Landmark probability threshold\n"
        << "  --display                Open an interactive window\n"
        << "  --no-display             Run headlessly (default)\n"
        << "  --validate               Check stable output invariants\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printHelp(argv[0]);
            return 0;
        }

        // imread returns an empty matrix for both missing and undecodable files.
        const cv::Mat frame =
            cv::imread(options.input.string(), cv::IMREAD_COLOR);
        if (frame.empty()) {
            throw std::runtime_error(
                "Could not read input image: " + options.input.string());
        }

        learnopencv::pose::PoseEstimator estimator(
            options.model, options.device);
        const learnopencv::pose::PoseResult result = estimator.infer(frame);

        cv::Mat output = frame.clone();
        const learnopencv::pose::DrawMetrics metrics =
            learnopencv::pose::draw(
                output, result, options.score_threshold);

        std::filesystem::create_directories(options.output_dir);
        const std::filesystem::path output_path =
            options.output_dir / "pose-image.jpg";
        if (!cv::imwrite(output_path.string(), output)) {
            throw std::runtime_error(
                "Could not write output image: " + output_path.string());
        }

        if (options.validate) {
            learnopencv::pose::validate(frame, result, metrics);
            std::cout
                << "VALIDATION PASSED: landmarks=33 visible="
                << metrics.visible_count
                << " edges=" << metrics.edge_count << '\n';
        }

        if (options.display) {
            cv::imshow("MediaPipe Pose", output);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        std::cout << "OpenCV version: " << CV_VERSION << '\n';
        std::cout
            << "POSE RESULT: confidence=" << result.confidence
            << " visible=" << metrics.visible_count
            << " edges=" << metrics.edge_count << '\n';
        std::cout << "Saved output: " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
