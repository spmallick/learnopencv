// Single-object tracking with OpenCV 5's six modern, non-legacy trackers.
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
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
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
    // MJPG in AVI is widely available in desktop OpenCV builds. The explicit
    // isOpened() check reports builds without a compatible writer backend.
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
    // Open explicitly and check the stream before emitting JSON. Without this
    // guard, a permissions or disk error could leave the user with no metrics
    // while the program still reported success.
    const fs::path metricsPath =
        *options.outputDir / ("metrics_" + options.trackerName + ".json");
    std::ofstream out(metricsPath, std::ios::out | std::ios::trunc);
    if (!out.is_open())
        throw std::runtime_error("Cannot open metrics file for writing: " +
                                 metricsPath.string());
    out << "{\n"
        << "  \"tracker\": \"" << options.trackerName << "\",\n"
        << "  \"opencv_version\": \"" << CV_VERSION << "\",\n"
        << "  \"frames\": " << metrics.frames << ",\n"
        << "  \"lost_frames\": " << metrics.lostFrames << ",\n"
        << "  \"mean_fps\": " << metrics.meanFps;
    if (metrics.meanIou)
        out << ",\n  \"mean_iou\": " << *metrics.meanIou
            << ",\n  \"success_rate\": " << *metrics.successRate;
    else
        // Normal runs have no ground truth. Explicit nulls keep the metrics
        // schema identical to Python and easier for downstream tools to parse.
        out << ",\n  \"mean_iou\": null"
            << ",\n  \"success_rate\": null";
    out << "\n}\n";
    // Flushing here turns delayed filesystem failures into a clean nonzero
    // program exit instead of allowing a truncated JSON file to pass silently.
    out.flush();
    if (!out)
        throw std::runtime_error("Cannot write metrics file: " +
                                 metricsPath.string());
}

// Parse exactly four comma-separated base-10 integers. std::from_chars does
// not skip whitespace and reports both overflow and trailing characters, so
// malformed values such as "10,20,30,40junk" cannot be accepted accidentally.
static cv::Rect parseBoundingBox(const std::string& text)
{
    std::array<int, 4> values{};
    std::size_t componentStart = 0;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        // Only the first three components may be followed by a comma. Requiring
        // the fourth component to end at text.size() rejects extra fields.
        const std::size_t componentEnd =
            index + 1 < values.size() ? text.find(',', componentStart)
                                      : text.size();
        if (componentEnd == std::string::npos || componentEnd == componentStart)
            throw std::runtime_error(
                "Invalid --bbox: expected exactly x,y,w,h integers");

        const char* const begin = text.data() + componentStart;
        const char* const end = text.data() + componentEnd;
        const auto result = std::from_chars(begin, end, values[index]);
        if (result.ec != std::errc{} || result.ptr != end)
            throw std::runtime_error(
                "Invalid --bbox: expected exactly x,y,w,h integers");

        componentStart = componentEnd + 1;
    }
    // A fourth comma would have been parsed as trailing junk in the final
    // component, but this explicit condition documents the exact contract.
    if (componentStart != text.size() + 1)
        throw std::runtime_error(
            "Invalid --bbox: expected exactly x,y,w,h integers");
    if (values[2] <= 0 || values[3] <= 0)
        throw std::runtime_error(
            "Invalid --bbox: width and height must be positive");
    return {values[0], values[1], values[2], values[3]};
}

// Check the rectangle without adding coordinates, which avoids integer
// overflow when a hostile or mistyped command-line value is near INT_MAX.
static void validateBoundingBox(const cv::Rect& box, const cv::Mat& frame)
{
    if (box.width <= 0 || box.height <= 0)
        throw std::runtime_error(
            "Invalid --bbox: width and height must be positive");
    if (box.x < 0 || box.y < 0 || box.x > frame.cols ||
        box.y > frame.rows || box.width > frame.cols - box.x ||
        box.height > frame.rows - box.y)
    {
        throw std::runtime_error(
            "Invalid --bbox: rectangle must lie inside the first frame");
    }
}

// Video files normally advertise their native frame rate. Cameras and unusual
// containers can report zero, NaN, or infinity, so use 30 FPS only when the
// source does not provide a usable positive value.
static double outputFrameRate(const cv::VideoCapture& capture)
{
    const double sourceFps = capture.get(cv::CAP_PROP_FPS);
    return std::isfinite(sourceFps) && sourceFps > 0.0 ? sourceFps : 30.0;
}

// OpenCV's parser validates conversions for declared keys but silently ignores
// undeclared keys. Scan only the option names here so a typo cannot fall
// through to a run with defaults; CommandLineParser still owns value parsing,
// aliases, help text, and conversion-error reporting.
static std::optional<std::string> findUnknownCommandLineToken(int argc,
                                                              char** argv)
{
    static const std::vector<std::string> knownNames = {
        "help", "h", "tracker", "input", "bbox", "models-dir",
        "output-dir", "max-frames", "no-display", "validate", "list"};
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument.size() < 2 || argument.front() != '-')
            return "Unexpected positional argument: " + argument;

        // CommandLineParser accepts either one or two leading dashes and an
        // optional '=value' suffix. Compare only the name before that suffix.
        const std::size_t nameStart =
            argument.size() > 2 && argument[1] == '-' ? 2 : 1;
        const std::size_t equals = argument.find('=', nameStart);
        const std::string name =
            argument.substr(nameStart, equals - nameStart);
        if (std::find(knownNames.begin(), knownNames.end(), name) ==
            knownNames.end())
        {
            return "Unknown option: " + argument;
        }
    }
    return std::nullopt;
}

// Core loop shared by normal runs and validation, mirroring Python's track().
// The caller supplies the already-read first frame. This guarantees that an
// interactive camera ROI initializes the tracker on the exact image the user
// saw, without attempting an impossible camera rewind.
static RunMetrics track(cv::Ptr<cv::Tracker> tracker, cv::VideoCapture& capture,
                        const cv::Mat& firstFrame, cv::Rect initBox,
                        const RunOptions& options,
                        const std::vector<cv::Rect>* groundTruth)
{
    // Reject an empty or out-of-frame ROI before any tracker accesses it.
    validateBoundingBox(initBox, firstFrame);
    // Teach the tracker from the pristine frame before adding annotations.
    tracker->init(firstFrame, initBox);
    // Prepare the optional annotated-video writer.
    cv::VideoWriter writer;
    if (options.outputDir)
    {
        // Create the directory explicitly instead of failing on open.
        fs::create_directories(*options.outputDir);
        const fs::path videoPath =
            *options.outputDir / ("tracked_" + options.trackerName + ".avi");
        writer.open(videoPath.string(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    outputFrameRate(capture), firstFrame.size());
        if (!writer.isOpened())
            throw std::runtime_error("Cannot write output video in " +
                                     options.outputDir->string());
    }
    // TickMeter times exactly the tracker updates for an honest FPS figure.
    cv::TickMeter meter;
    RunMetrics metrics;
    metrics.frames = 1;  // the init frame counts toward the total
    std::vector<double> ious;
    // The initialized frame is a processed frame too. Annotating and writing
    // it here makes output frame counts agree with metrics.frames and leaves a
    // readable one-frame AVI when --max-frames=1 is requested.
    cv::Mat frame = firstFrame.clone();
    cv::rectangle(frame, initBox, {0, 255, 0}, 2);
    cv::putText(frame, options.trackerName + "  initialized", {20, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, {50, 170, 50}, 2);
    if (writer.isOpened())
        writer.write(frame);
    bool userStopped = false;
    if (!options.noDisplay)
    {
        cv::imshow("Tracking", frame);
        userStopped = (cv::waitKey(1) & 0xFF) == 27;
    }
    while (!userStopped)
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
        // Count this frame before any interactive early exit; it has already
        // been read, tracked, scored, and will be written below.
        ++metrics.frames;
        if (writer.isOpened())
            writer.write(frame);
        // Display unless headless; ESC exits an interactive session.
        if (!options.noDisplay)
        {
            cv::imshow("Tracking", frame);
            if ((cv::waitKey(1) & 0xFF) == 27)
                break;
        }
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
    // Retrieve every typed option before check(): conversion errors are added
    // to the parser's error list by get() and printed together below.
    const std::string trackerArgument = parser.get<std::string>("tracker");
    const std::string inputArgument = parser.get<std::string>("input");
    const std::string bboxArgument = parser.get<std::string>("bbox");
    const std::string modelsDirArgument =
        parser.get<std::string>("models-dir");
    const std::string outputDirArgument =
        parser.get<std::string>("output-dir");
    const int maxFramesArgument = parser.get<int>("max-frames");
    // OpenCV 4.14 and 5.0 both ignore undeclared keys internally, so close that
    // gap explicitly while retaining CommandLineParser for declared options.
    if (const auto unknown = findUnknownCommandLineToken(argc, argv))
    {
        std::cerr << "error: " << *unknown << std::endl;
        return 2;
    }
    if (!parser.check())
    {
        // CommandLineParser formats all accumulated errors consistently.
        parser.printErrors();
        return 2;
    }
    if (maxFramesArgument < 0)
    {
        std::cerr << "error: --max-frames must be zero or positive"
                  << std::endl;
        return 2;
    }
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Resolve the models directory: CLI flag first, compiled default second.
    const fs::path modelsDir = modelsDirArgument.empty()
        ? fs::path(MODELS_DIR)
        : fs::path(modelsDirArgument);
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
        options.trackerName = trackerArgument;
        options.noDisplay = parser.has("no-display");
        options.maxFrames = maxFramesArgument;
        if (!outputDirArgument.empty())
            options.outputDir = fs::path(outputDirArgument);
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
            // Read once here so track() receives exactly the initialization
            // frame and begins update() calls on the following frame.
            cv::Mat firstFrame;
            if (!capture.read(firstFrame))
                throw std::runtime_error("Input has no frames");
            const RunMetrics metrics =
                track(tracker, capture, firstFrame, truth.front(), options, &truth);
            writeMetricsJson(metrics, options);
            // Print the metrics and the unambiguous CI marker.
            std::cout << "mean_iou=" << metrics.meanIou.value_or(0.0)
                      << " success_rate=" << metrics.successRate.value_or(0.0)
                      << " mean_fps=" << metrics.meanFps << "\n";
            // A short high-quality prefix is not a complete regression.
            // Require all generated frames as well as the quality thresholds.
            const bool passed =
                metrics.frames == static_cast<int>(truth.size()) &&
                metrics.meanIou &&
                *metrics.meanIou >= kValidateMeanIou &&
                *metrics.successRate >= kValidateSuccessRate;
            std::cout << (passed ? "VALIDATION PASSED" : "VALIDATION FAILED")
                      << std::endl;
            return passed ? 0 : 1;
        }
        // Normal mode: open the requested video file or camera index.
        const std::string input = inputArgument;
        const bool isCameraIndex =
            !input.empty() &&
            input.find_first_not_of("0123456789") == std::string::npos;
        cv::VideoCapture capture;
        if (isCameraIndex) capture.open(std::stoi(input));
        else capture.open(input);
        if (!capture.isOpened())
            throw std::runtime_error("Cannot open input: " + input);
        // Capture the initialization image once. For live cameras it cannot be
        // replayed, so both ROI selection and tracker initialization use it.
        cv::Mat firstFrame;
        if (!capture.read(firstFrame))
            throw std::runtime_error("Input has no frames");
        // Determine the initial box: parsed from --bbox, or drawn by hand.
        cv::Rect initBox;
        const std::string bbox = bboxArgument;
        if (!bbox.empty())
        {
            initBox = parseBoundingBox(bbox);
        }
        else
        {
            if (options.noDisplay)
                throw std::runtime_error("--bbox is required when --no-display is set");
            // Let the user draw the box on the first frame interactively.
            initBox = cv::selectROI("Select object", firstFrame, true);
            cv::destroyWindow("Select object");
        }
        const RunMetrics metrics =
            track(tracker, capture, firstFrame, initBox, options, nullptr);
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
