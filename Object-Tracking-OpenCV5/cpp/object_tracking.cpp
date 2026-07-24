// Single-object tracking with every tracker that ships with OpenCV 5.
//
// This is the C++ twin of ../python/object_tracking.py. One executable
// drives all six trackers:
//
//     classical (no model files):  MIL, KCF, CSRT
//     deep-learning (ONNX files):  DaSiamRPN, NanoTrack, TrackerVit
//
// The same source builds unchanged against OpenCV 4.x (4.9+) and OpenCV 5.x.
// KCF and CSRT come from the opencv_contrib "tracking" module; the code
// detects its absence at compile time and degrades gracefully instead of
// failing to build on a main-only OpenCV installation.
//
// Examples:
//     ./object_tracking --list
//     ./object_tracking --tracker=vittrack --input=video.mp4
//     ./object_tracking --tracker=mil --validate --no-display

// Core matrix types, cv::Rect, cv::TickMeter, and cv::CommandLineParser.
#include <opencv2/core.hpp>
// Reports which modules this OpenCV build contains (HAVE_OPENCV_TRACKING).
#include <opencv2/opencv_modules.hpp>
// Drawing primitives (rectangle, putText) and image resizing.
#include <opencv2/imgproc.hpp>
// VideoCapture and VideoWriter for file, camera, and clip generation.
#include <opencv2/videoio.hpp>
// The main-module trackers: MIL, DaSiamRPN, Nano, Vit (GOTURN in 4.x only).
#include <opencv2/video/tracking.hpp>
// The GUI calls are needed only for interactive display, but highgui is a
// hard OpenCV dependency of this example so we include it unconditionally.
#include <opencv2/highgui.hpp>

// The contrib tracking module supplies KCF and CSRT when present.
#ifdef HAVE_OPENCV_TRACKING
#include <opencv2/tracking.hpp>
#endif

// Standard library: filesystem for path handling, containers, formatting.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Constants mirrored from the Python implementation so that the synthetic
// validation clip and the pass criteria stay in lockstep across languages.
// ---------------------------------------------------------------------------

// Frame geometry and length of the synthetic validation clip.
constexpr int kSynthWidth = 640;
constexpr int kSynthHeight = 360;
constexpr int kSynthFrames = 80;
constexpr int kSynthTarget = 64;   // side length of the moving square
constexpr int kSynthSeed = 7;      // fixed seed keeps the clip deterministic

// Validation thresholds: loose on purpose. They catch a broken tracker or a
// broken API call, not small quality differences between algorithms.
constexpr double kValidateMeanIou = 0.45;
constexpr double kValidateSuccessIou = 0.30;
constexpr double kValidateSuccessRate = 0.90;

// MODELS_DIR is injected by CMake as the absolute path of ../models so the
// executable finds the shared ONNX files regardless of the current directory.
#ifndef MODELS_DIR
#define MODELS_DIR "../models"
#endif

// ---------------------------------------------------------------------------
// Tracker factory
// ---------------------------------------------------------------------------

// Build the requested tracker, or return an empty pointer with a reason the
// caller can show. Missing contrib and missing model files are different
// user problems, so the message distinguishes them.
static cv::Ptr<cv::Tracker> createTracker(const std::string& name,
                                          const fs::path& modelsDir,
                                          std::string& whyNot)
{
    // Small helper: true when every listed model file exists on disk.
    const auto allExist = [&](std::initializer_list<const char*> files) {
        for (const char* file : files)
            if (!fs::exists(modelsDir / file)) return false;
        return true;
    };

    if (name == "mil")
    {
        // MIL lives in the main video module of both 4.x and 5.x.
        return cv::TrackerMIL::create();
    }
    if (name == "kcf" || name == "csrt")
    {
#ifdef HAVE_OPENCV_TRACKING
        // KCF and CSRT come from opencv_contrib's tracking module.
        if (name == "kcf") return cv::TrackerKCF::create();
        return cv::TrackerCSRT::create();
#else
        whyNot = name + " requires an OpenCV build with the opencv_contrib "
                        "tracking module";
        return nullptr;
#endif
    }
    if (name == "dasiamrpn")
    {
        // DaSiamRPN needs a backbone network plus two correlation kernels.
        if (!allExist({"dasiamrpn_model.onnx", "dasiamrpn_kernel_cls1.onnx",
                       "dasiamrpn_kernel_r1.onnx"}))
        {
            whyNot = "dasiamrpn model files missing; run download_models.py";
            return nullptr;
        }
        cv::TrackerDaSiamRPN::Params params;
        params.model = (modelsDir / "dasiamrpn_model.onnx").string();
        params.kernel_cls1 = (modelsDir / "dasiamrpn_kernel_cls1.onnx").string();
        params.kernel_r1 = (modelsDir / "dasiamrpn_kernel_r1.onnx").string();
        return cv::TrackerDaSiamRPN::create(params);
    }
    if (name == "nanotrack")
    {
        // NanoTrack splits its ~2 MB network into a backbone and neck+head.
        if (!allExist({"nanotrack_backbone_sim.onnx", "nanotrack_head_sim.onnx"}))
        {
            whyNot = "nanotrack model files missing; run download_models.py";
            return nullptr;
        }
        cv::TrackerNano::Params params;
        params.backbone = (modelsDir / "nanotrack_backbone_sim.onnx").string();
        params.neckhead = (modelsDir / "nanotrack_head_sim.onnx").string();
        return cv::TrackerNano::create(params);
    }
    if (name == "vittrack")
    {
        // TrackerVit uses a single sub-megabyte transformer ONNX file.
        if (!allExist({"object_tracking_vittrack_2023sep.onnx"}))
        {
            whyNot = "vittrack model file missing; run download_models.py";
            return nullptr;
        }
        cv::TrackerVit::Params params;
        params.net = (modelsDir / "object_tracking_vittrack_2023sep.onnx").string();
        return cv::TrackerVit::create(params);
    }
    whyNot = "unknown tracker name: " + name;
    return nullptr;
}

// The canonical tracker order used by --list and the help text.
static const std::vector<std::string> kTrackerNames =
    {"mil", "kcf", "csrt", "dasiamrpn", "nanotrack", "vittrack"};

// ---------------------------------------------------------------------------
// Synthetic validation clip
// ---------------------------------------------------------------------------

// Ground-truth trajectory: the same Lissajous-style path as the Python
// version, topping out near 10 px/frame so every tracker can follow.
static cv::Rect groundTruthBox(int frameIndex)
{
    const double phase = static_cast<double>(frameIndex) / kSynthFrames;
    const int x = static_cast<int>(
        (kSynthWidth - kSynthTarget - 40) * 0.5 *
            (1.0 + std::sin(CV_PI * phase - CV_PI / 2)) + 20);
    const int y = static_cast<int>(
        (kSynthHeight - kSynthTarget - 40) * 0.5 *
            (1.0 + std::sin(2 * CV_PI * phase)) + 20);
    return {x, y, kSynthTarget, kSynthTarget};
}

// Write the deterministic clip of a textured square moving over a noisy
// background and return the per-frame ground-truth boxes. cv::RNG and
// numpy differ, so the pixels are not byte-identical to Python's clip, but
// the geometry, path, and difficulty are the same by construction.
static std::vector<cv::Rect> makeSyntheticVideo(const fs::path& path)
{
    // Seeded RNG makes every generation of the clip identical.
    cv::RNG rng(kSynthSeed);
    // Background: horizontal gradient with per-channel uniform noise.
    cv::Mat background(kSynthHeight, kSynthWidth, CV_8UC3);
    for (int row = 0; row < kSynthHeight; ++row)
    {
        for (int col = 0; col < kSynthWidth; ++col)
        {
            // Gradient term matches Python's linspace(60, 120) horizontally.
            const int gradient = 60 + col * 60 / (kSynthWidth - 1);
            auto& pixel = background.at<cv::Vec3b>(row, col);
            for (int channel = 0; channel < 3; ++channel)
                pixel[channel] = static_cast<unsigned char>(
                    std::min(255, gradient + static_cast<int>(rng.uniform(0, 40))));
        }
    }
    // Target: an 8x8 grid of random bright colors, upscaled with nearest
    // neighbor so each cell stays a crisp, trackable block.
    cv::Mat targetSmall(8, 8, CV_8UC3);
    rng.fill(targetSmall, cv::RNG::UNIFORM, 64, 255);
    cv::Mat target;
    cv::resize(targetSmall, target, {kSynthTarget, kSynthTarget}, 0, 0,
               cv::INTER_NEAREST);
    // MJPG in AVI is compiled into every OpenCV binary; no external codec.
    cv::VideoWriter writer(path.string(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           30.0, {kSynthWidth, kSynthHeight});
    if (!writer.isOpened())
        throw std::runtime_error("Cannot open video writer for " + path.string());
    std::vector<cv::Rect> boxes;
    boxes.reserve(kSynthFrames);
    for (int index = 0; index < kSynthFrames; ++index)
    {
        // Compose each frame: background copy, then paste the target.
        const cv::Rect box = groundTruthBox(index);
        cv::Mat frame = background.clone();
        target.copyTo(frame(box));
        writer.write(frame);
        boxes.push_back(box);
    }
    return boxes;
}

// Intersection-over-union of two rectangles; cv::Rect supports operator&.
static double iou(const cv::Rect& a, const cv::Rect& b)
{
    const double intersection = static_cast<double>((a & b).area());
    const double unionArea = a.area() + b.area() - intersection;
    return unionArea > 0.0 ? intersection / unionArea : 0.0;
}

// ---------------------------------------------------------------------------
// Tracking loop
// ---------------------------------------------------------------------------

// Options shared by the normal and validation code paths.
struct RunOptions
{
    std::string trackerName;          // which tracker to run
    std::optional<fs::path> outputDir;// annotated video + metrics target
    bool noDisplay = false;           // headless mode for tests and CI
    int maxFrames = 0;                // 0 means process the whole stream
};

// Result metrics; written as JSON so tests and benchmarks can consume them.
struct RunMetrics
{
    int frames = 0;                   // frames processed including init
    int lostFrames = 0;               // frames where update() reported loss
    double meanFps = 0.0;             // average tracker-update FPS
    std::optional<double> meanIou;    // vs ground truth when validating
    std::optional<double> successRate;// fraction of frames with IoU > 0.30
};

// Serialize metrics as a small JSON object, mirroring the Python output.
static void writeMetricsJson(const RunMetrics& metrics, const RunOptions& options)
{
    if (!options.outputDir) return;
    std::ofstream out(*options.outputDir / ("metrics_" + options.trackerName + ".json"));
    out << "{\n"
        << "  \"tracker\": \"" << options.trackerName << "\",\n"
        << "  \"opencv_version\": \"" << CV_VERSION << "\",\n"
        << "  \"frames\": " << metrics.frames << ",\n"
        << "  \"lost_frames\": " << metrics.lostFrames << ",\n"
        << "  \"mean_fps\": " << metrics.meanFps;
    if (metrics.meanIou)
        out << ",\n  \"mean_iou\": " << *metrics.meanIou
            << ",\n  \"success_rate\": " << *metrics.successRate;
    out << "\n}\n";
}

// Core loop shared by normal runs and validation, mirroring Python's track().
static RunMetrics track(cv::Ptr<cv::Tracker> tracker, cv::VideoCapture& capture,
                        cv::Rect initBox, const RunOptions& options,
                        const std::vector<cv::Rect>* groundTruth)
{
    // Read the first frame and teach the tracker the target's appearance.
    cv::Mat frame;
    if (!capture.read(frame))
        throw std::runtime_error("Input has no frames");
    tracker->init(frame, initBox);
    // Prepare the optional annotated-video writer.
    cv::VideoWriter writer;
    if (options.outputDir)
    {
        // Create the directory explicitly instead of failing on open.
        fs::create_directories(*options.outputDir);
        const fs::path videoPath =
            *options.outputDir / ("tracked_" + options.trackerName + ".avi");
        writer.open(videoPath.string(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                    frame.size());
        if (!writer.isOpened())
            throw std::runtime_error("Cannot write output video in " +
                                     options.outputDir->string());
    }
    // TickMeter times exactly the tracker updates for an honest FPS figure.
    cv::TickMeter meter;
    RunMetrics metrics;
    metrics.frames = 1;  // the init frame counts toward the total
    std::vector<double> ious;
    while (true)
    {
        // Honor the optional frame budget used by quick test runs.
        if (options.maxFrames > 0 && metrics.frames >= options.maxFrames)
            break;
        if (!capture.read(frame))
            break;  // end of stream
        cv::Rect box;
        // Time only the update call, not drawing or file I/O.
        meter.start();
        const bool found = tracker->update(frame, box);
        meter.stop();
        if (found)
        {
            // Draw the tracked box in green.
            cv::rectangle(frame, box, {0, 255, 0}, 2);
        }
        else
        {
            // Announce failure on the frame and count it.
            ++metrics.lostFrames;
            cv::putText(frame, "tracking failure", {20, 60},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
        }
        // Score against ground truth while validating.
        if (groundTruth != nullptr &&
            metrics.frames < static_cast<int>(groundTruth->size()))
        {
            const cv::Rect truth = (*groundTruth)[metrics.frames];
            ious.push_back(found ? iou(box, truth) : 0.0);
        }
        // Overlay the running-average FPS of tracker updates.
        cv::putText(frame,
                    options.trackerName + cv::format("  %5.1f FPS", meter.getFPS()),
                    {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {50, 170, 50}, 2);
        if (writer.isOpened())
            writer.write(frame);
        // Display unless headless; ESC exits an interactive session.
        if (!options.noDisplay)
        {
            cv::imshow("Tracking", frame);
            if ((cv::waitKey(1) & 0xFF) == 27)
                break;
        }
        ++metrics.frames;
    }
    // Aggregate the metrics after the loop.
    metrics.meanFps = meter.getFPS();
    if (!ious.empty())
    {
        double sum = 0.0;
        int successes = 0;
        for (const double value : ious)
        {
            sum += value;
            if (value > kValidateSuccessIou) ++successes;
        }
        metrics.meanIou = sum / static_cast<double>(ious.size());
        metrics.successRate =
            static_cast<double>(successes) / static_cast<double>(ious.size());
    }
    return metrics;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // cv::CommandLineParser keeps the CLI declaration compact and readable.
    const std::string keys =
        "{help h        |          | print this help }"
        "{tracker       | vittrack | mil kcf csrt dasiamrpn nanotrack vittrack }"
        "{input         | 0        | video path or camera index }"
        "{bbox          |          | initial box as x,y,w,h }"
        "{models-dir    |          | directory holding the ONNX models }"
        "{output-dir    |          | write annotated video and metrics here }"
        "{max-frames    | 0        | stop after this many frames (0 = all) }"
        "{no-display    |          | run headless: no windows, no waitKey }"
        "{validate      |          | run the synthetic-clip regression check }"
        "{list          |          | report tracker availability and exit }";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Single-object tracking with the OpenCV 5 tracker lineup");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Resolve the models directory: CLI flag first, compiled default second.
    const fs::path modelsDir = parser.get<std::string>("models-dir").empty()
        ? fs::path(MODELS_DIR)
        : fs::path(parser.get<std::string>("models-dir"));
    // --list probes every tracker and reports availability, like Python's
    // --list-trackers, then exits successfully.
    if (parser.has("list"))
    {
        std::cout << "OpenCV version: " << CV_VERSION << "\n";
        for (const std::string& name : kTrackerNames)
        {
            std::string whyNot;
            cv::Ptr<cv::Tracker> probe;
            try { probe = createTracker(name, modelsDir, whyNot); }
            catch (const cv::Exception&) { probe = nullptr; }
            std::cout << "  " << name << (probe ? "  available" : "  NOT available")
                      << (whyNot.empty() ? "" : "  (" + whyNot + ")") << "\n";
        }
        return 0;
    }
    try
    {
        // Gather the shared options once.
        RunOptions options;
        options.trackerName = parser.get<std::string>("tracker");
        options.noDisplay = parser.has("no-display");
        options.maxFrames = parser.get<int>("max-frames");
        if (!parser.get<std::string>("output-dir").empty())
            options.outputDir = fs::path(parser.get<std::string>("output-dir"));
        // Build the tracker or explain exactly why it cannot be built.
        std::string whyNot;
        cv::Ptr<cv::Tracker> tracker =
            createTracker(options.trackerName, modelsDir, whyNot);
        if (!tracker)
            throw std::runtime_error(whyNot);
        if (parser.has("validate"))
        {
            // Validation mode: synthesize the clip, then require the tracker
            // to stay on target within the documented thresholds.
            const fs::path outputDir =
                options.outputDir.value_or(fs::path("outputs"));
            fs::create_directories(outputDir);
            options.outputDir = outputDir;
            const fs::path clipPath = outputDir / "synthetic_clip.avi";
            const std::vector<cv::Rect> truth = makeSyntheticVideo(clipPath);
            cv::VideoCapture capture(clipPath.string());
            if (!capture.isOpened())
                throw std::runtime_error("Cannot open input: " + clipPath.string());
            const RunMetrics metrics =
                track(tracker, capture, truth.front(), options, &truth);
            writeMetricsJson(metrics, options);
            // Print the metrics and the unambiguous CI marker.
            std::cout << "mean_iou=" << metrics.meanIou.value_or(0.0)
                      << " success_rate=" << metrics.successRate.value_or(0.0)
                      << " mean_fps=" << metrics.meanFps << "\n";
            const bool passed = metrics.meanIou &&
                                *metrics.meanIou >= kValidateMeanIou &&
                                *metrics.successRate >= kValidateSuccessRate;
            std::cout << (passed ? "VALIDATION PASSED" : "VALIDATION FAILED")
                      << std::endl;
            return passed ? 0 : 1;
        }
        // Normal mode: open the requested video file or camera index.
        const std::string input = parser.get<std::string>("input");
        const bool isCameraIndex =
            !input.empty() &&
            input.find_first_not_of("0123456789") == std::string::npos;
        cv::VideoCapture capture;
        if (isCameraIndex) capture.open(std::stoi(input));
        else capture.open(input);
        if (!capture.isOpened())
            throw std::runtime_error("Cannot open input: " + input);
        // Determine the initial box: parsed from --bbox, or drawn by hand.
        cv::Rect initBox;
        const std::string bbox = parser.get<std::string>("bbox");
        if (!bbox.empty())
        {
            // Parse the "x,y,w,h" string with sscanf-style extraction.
            if (std::sscanf(bbox.c_str(), "%d,%d,%d,%d", &initBox.x, &initBox.y,
                            &initBox.width, &initBox.height) != 4)
                throw std::runtime_error("Cannot parse --bbox=" + bbox);
        }
        else
        {
            if (options.noDisplay)
                throw std::runtime_error("--bbox is required when --no-display is set");
            // Let the user draw the box on the first frame interactively.
            cv::Mat first;
            if (!capture.read(first))
                throw std::runtime_error("Input has no frames");
            initBox = cv::selectROI("Select object", first, true);
            cv::destroyWindow("Select object");
            // Rewind file inputs so tracking starts at the first frame again.
            capture.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
        const RunMetrics metrics = track(tracker, capture, initBox, options, nullptr);
        writeMetricsJson(metrics, options);
        std::cout << "frames=" << metrics.frames
                  << " lost=" << metrics.lostFrames
                  << " mean_fps=" << metrics.meanFps << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        // One readable error line beats an unhandled-exception abort.
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
