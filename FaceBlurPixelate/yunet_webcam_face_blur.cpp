/*
YuNet Webcam Face Anonymizer (C++ Tutorial Version)

This example demonstrates a full real-time pipeline with OpenCV:
1) Open webcam stream.
2) Load YuNet neural face detector (FaceDetectorYN) from an ONNX model.
3) Detect faces on each frame.
4) Apply one of two anonymization methods on each detected face:
   - Gaussian blur
   - Pixelation / mosaic effect
5) Show the processed video feed live.

Why this file is heavily commented:
- The goal is to be instructional, so each major section explains not only
  "what" it does, but also "why" it is done this way.

Build example:
  g++ -std=c++17 yunet_webcam_face_blur.cpp -o yunet_webcam_face_blur `pkg-config --cflags --libs opencv4`

Run example:
  ./yunet_webcam_face_blur --model face_detection_yunet_2023mar.onnx --mode pixelate
*/

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/videoio.hpp>

// Alias to reduce verbosity when using std::filesystem APIs.
namespace fs = std::filesystem;

/*
Configuration for the app.

This struct mirrors the CLI options so parameters are centralized and easy to
pass around.
*/
struct AppConfig {
    std::string model_path;       // Path to YuNet ONNX model; empty means auto-search.
    int camera_index = 0;         // Webcam index (0 is usually default camera).
    float score_threshold = 0.9f; // Minimum confidence for detections.
    float nms_threshold = 0.3f;   // NMS threshold to merge overlapping detections.
    int top_k = 5000;             // Max raw detections before NMS.
    bool show_score = false;      // If true, draw score text + rectangle.
    std::string mode = "blur";    // Anonymization mode: "blur" or "pixelate".
    int pixel_block_size = 16;    // Pixelation strength control.
};

// If --model is not provided, we try these filenames automatically.
static const std::vector<std::string> kDefaultModelCandidates = {
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
};

// Print command-line help.
static void printUsage(const char* prog) {
    std::cout
        << "YuNet Webcam Face Anonymizer (C++)\n\n"
        << "Usage:\n  " << prog
        << " [--model <path>] [--camera <idx>] [--score-threshold <float>]\n"
        << "      [--nms-threshold <float>] [--top-k <int>] [--show-score]\n"
        << "      [--mode blur|pixelate] [--pixel-block-size <int>]\n\n"
        << "Examples:\n"
        << "  " << prog << " --model face_detection_yunet_2023mar.onnx --mode blur\n"
        << "  " << prog << " --model face_detection_yunet_2023mar.onnx --mode pixelate --pixel-block-size 20\n";
}

/*
Find a value for args in the form:
  --key value

Returns true and writes out_value if key is found with a following token.
*/
static bool getArgValue(
    int argc,
    char** argv,
    const std::string& key,
    std::string& out_value
) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == key) {
            out_value = argv[i + 1];
            return true;
        }
    }
    return false;
}

// Return true if a boolean flag (e.g. --show-score) is present.
static bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

/*
Parse all CLI args into AppConfig.

Behavior notes:
- --help/-h prints usage and returns false (caller exits without error).
- Mode is validated to only allow blur/pixelate.
- Pixel block size is clamped to at least 1.
*/
static bool parseArgs(int argc, char** argv, AppConfig& cfg) {
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(argv[0]);
        return false;
    }

    std::string value;

    // Optional model path.
    if (getArgValue(argc, argv, "--model", value)) {
        cfg.model_path = value;
    }

    // Camera index.
    if (getArgValue(argc, argv, "--camera", value)) {
        cfg.camera_index = std::stoi(value);
    }

    // Detector confidence threshold.
    if (getArgValue(argc, argv, "--score-threshold", value)) {
        cfg.score_threshold = std::stof(value);
    }

    // Detector NMS threshold.
    if (getArgValue(argc, argv, "--nms-threshold", value)) {
        cfg.nms_threshold = std::stof(value);
    }

    // Detector top-k.
    if (getArgValue(argc, argv, "--top-k", value)) {
        cfg.top_k = std::stoi(value);
    }

    // Processing mode.
    if (getArgValue(argc, argv, "--mode", value)) {
        cfg.mode = value;
    }

    // Pixelation strength.
    if (getArgValue(argc, argv, "--pixel-block-size", value)) {
        cfg.pixel_block_size = std::stoi(value);
    }

    // Bool flag.
    cfg.show_score = hasFlag(argc, argv, "--show-score");

    // Validate mode value explicitly.
    if (cfg.mode != "blur" && cfg.mode != "pixelate") {
        std::cerr << "Invalid --mode value: " << cfg.mode
                  << " (expected blur or pixelate)\n";
        return false;
    }

    // Keep this safe to avoid divide-by-zero in pixelation logic.
    if (cfg.pixel_block_size < 1) {
        cfg.pixel_block_size = 1;
    }

    return true;
}

/*
Resolve model path.

Search order:
1) If user passed --model, validate and use it.
2) Current working directory + default YuNet names.
3) Executable directory + default YuNet names.

Throws std::runtime_error if model cannot be found.
*/
static std::string resolveModelPath(const AppConfig& cfg, const char* argv0) {
    // Highest priority: explicit user path.
    if (!cfg.model_path.empty()) {
        fs::path p(cfg.model_path);
        p = fs::absolute(p);
        if (fs::is_regular_file(p)) {
            return p.string();
        }
        throw std::runtime_error("Model not found: " + p.string());
    }

    // Try current working directory.
    fs::path cwd = fs::current_path();
    for (const auto& name : kDefaultModelCandidates) {
        fs::path candidate = cwd / name;
        if (fs::is_regular_file(candidate)) {
            return candidate.string();
        }
    }

    // Try executable directory.
    fs::path exe_dir = fs::absolute(fs::path(argv0)).parent_path();
    for (const auto& name : kDefaultModelCandidates) {
        fs::path candidate = exe_dir / name;
        if (fs::is_regular_file(candidate)) {
            return candidate.string();
        }
    }

    throw std::runtime_error(
        "YuNet model not found. Pass --model <path> or place "
        "face_detection_yunet_2023mar.onnx / face_detection_yunet_2022mar.onnx "
        "in current directory."
    );
}

/*
Clamp a rectangle to frame bounds.

Face boxes can sometimes be partially outside the frame,
especially near image borders. This function protects ROI slicing.
*/
static bool clampRect(const cv::Rect& in, int frame_w, int frame_h, cv::Rect& out) {
    int x1 = std::max(0, in.x);
    int y1 = std::max(0, in.y);
    int x2 = std::min(frame_w, in.x + in.width);
    int y2 = std::min(frame_h, in.y + in.height);

    if (x2 <= x1 || y2 <= y1) {
        return false;
    }

    out = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    return true;
}

/*
Apply Gaussian blur to a face region.

Kernel size is adapted based on face size so the effect is reasonably consistent
whether the face is small (far) or large (near).
*/
static void blurFaceRegion(cv::Mat& frame, const cv::Rect& rect) {
    cv::Mat roi = frame(rect);
    if (roi.empty()) {
        return;
    }

    int min_dim = std::min(rect.width, rect.height);

    // Base kernel from face size.
    int kernel = std::max(15, min_dim / 3);

    // Gaussian kernels must be odd values.
    if (kernel % 2 == 0) {
        kernel += 1;
    }

    cv::GaussianBlur(roi, roi, cv::Size(kernel, kernel), 0);
}

/*
Apply pixelation (mosaic) to a face region.

Classic trick:
1) Downscale aggressively -> detail removed.
2) Upscale with INTER_NEAREST -> visible square blocks.
*/
static void pixelateFaceRegion(cv::Mat& frame, const cv::Rect& rect, int block_size) {
    cv::Mat roi = frame(rect);
    if (roi.empty()) {
        return;
    }

    // Safety clamp.
    int safe_block = std::max(1, block_size);

    // Compute temporary low-res size.
    // Bigger block size => smaller temp image => stronger pixel effect.
    int small_w = std::max(1, roi.cols / safe_block);
    int small_h = std::max(1, roi.rows / safe_block);

    cv::Mat temp;

    // Shrink (loses detail).
    cv::resize(roi, temp, cv::Size(small_w, small_h), 0.0, 0.0, cv::INTER_LINEAR);

    // Expand with nearest-neighbor (keeps hard square blocks).
    cv::resize(temp, roi, roi.size(), 0.0, 0.0, cv::INTER_NEAREST);
}

int main(int argc, char** argv) {
    try {
        AppConfig cfg;

        // Parse CLI and potentially print help.
        if (!parseArgs(argc, argv, cfg)) {
            return 0;
        }

        // Locate ONNX model file.
        const std::string model_path = resolveModelPath(cfg, argv[0]);

        // Open webcam stream.
        cv::VideoCapture cap(cfg.camera_index);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open webcam index " << cfg.camera_index << "\n";
            return 1;
        }

        // Read one frame first so we know actual resolution.
        // FaceDetectorYN requires an explicit input size at creation.
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Cannot read initial frame from webcam\n";
            return 1;
        }

        // Create YuNet detector instance.
        // API:
        // FaceDetectorYN::create(model, config, input_size, score_thresh, nms_thresh, top_k)
        cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
            model_path,
            "",
            frame.size(),
            cfg.score_threshold,
            cfg.nms_threshold,
            cfg.top_k
        );

        std::cout << "Using model: " << model_path << "\n";
        std::cout << "Mode: " << cfg.mode << "\n";
        std::cout << "Press 'q' or ESC to quit\n";

        while (true) {
            // Grab frame.
            if (!cap.read(frame) || frame.empty()) {
                break;
            }

            // Keep detector input size synced with frame size.
            // Important if capture resolution changes.
            detector->setInputSize(frame.size());

            // Run detector.
            // Output matrix format: N x 15 float values per detected face:
            // [x, y, w, h, l0x, l0y, l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y, score]
            cv::Mat faces;
            detector->detect(frame, faces);

            // Iterate over all detections.
            for (int i = 0; i < faces.rows; ++i) {
                const cv::Mat row = faces.row(i);

                int x = static_cast<int>(row.at<float>(0));
                int y = static_cast<int>(row.at<float>(1));
                int w = static_cast<int>(row.at<float>(2));
                int h = static_cast<int>(row.at<float>(3));
                float score = row.at<float>(14);

                // Ensure rectangle is valid and inside frame bounds.
                cv::Rect clamped;
                if (!clampRect(cv::Rect(x, y, w, h), frame.cols, frame.rows, clamped)) {
                    continue;
                }

                // Apply selected anonymization mode.
                if (cfg.mode == "pixelate") {
                    pixelateFaceRegion(frame, clamped, cfg.pixel_block_size);
                } else {
                    blurFaceRegion(frame, clamped);
                }

                // Optional score display to help tune thresholds.
                if (cfg.show_score) {
                    cv::rectangle(frame, clamped, cv::Scalar(40, 255, 40), 2);
                    cv::putText(
                        frame,
                        cv::format("%.2f", score),
                        cv::Point(clamped.x, std::max(0, clamped.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(40, 255, 40),
                        2,
                        cv::LINE_AA
                    );
                }
            }

            // Display processed frame.
            cv::imshow("YuNet Face Blur/Pixelate (C++)", frame);

            // waitKey(1) also handles UI event pumping for HighGUI windows.
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') {
                break;
            }
        }

        // Cleanup.
        cap.release();
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& e) {
        // Centralized error handling for argument/model/path issues.
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
