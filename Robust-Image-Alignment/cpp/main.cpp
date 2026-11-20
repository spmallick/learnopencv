// Native command-line entry point for the notebook's alignment operations.
#include "alignment.hpp"
#include "feature_matcher.hpp"
#include "workflows.hpp"

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ROBUST_ALIGNMENT_PROJECT_DIR
#define ROBUST_ALIGNMENT_PROJECT_DIR "."
#endif

namespace {
namespace fs = std::filesystem;
namespace alignment = robust_alignment;

struct Arguments {
    alignment::WorkflowOptions workflow;
    fs::path reference;
    fs::path moving;
    fs::path correspondences;
    std::string method = "lightglue";
    std::string estimator = "magsac";
    bool demo_all = false;
    bool benchmark = false;
    bool synthetic = false;
    bool refine = false;
    bool dump_features = false;
    bool validate = false;
    bool help = false;
};

void show_help() {
    std::cout <<
        "Robust image alignment: native C++17 / OpenCV 5.1 / ONNX Runtime CPU\n\n"
        "Pair: --reference IMAGE --moving IMAGE [options]\n"
        "Notebook workflows: --demo-all | --benchmark | --synthetic-study\n\n"
        "  --method sift|aliked-nn|lightglue   Matching route (default lightglue)\n"
        "  --estimator ransac|magsac          Pair estimator (default magsac)\n"
        "  --model-dir PATH                  Exported ONNX model directory\n"
        "  --project-dir PATH                Relocate the companion source/assets\n"
        "  --assets-dir PATH                 Override downloaded sample directory\n"
        "  --output-dir PATH                 Results (default alignment-output-cpp)\n"
        "  --max-keypoints N                 Feature budget (default 2048)\n"
        "  --resize N                        Longest-side cap; 0 keeps original size\n"
        "  --threshold PIXELS                Original reference-pixel threshold\n"
        "  --min-inliers N                   Acceptance minimum (default 12)\n"
        "  --repeats N --study-seeds N        Benchmark / synthetic repetitions\n"
        "  --refine-ecc                      Save an optional refinement candidate\n"
        "  --dump-features                   Save points/descriptors for inspection\n"
        "  --correspondences JSON            Estimate supplied paired points directly\n"
        "  --device cpu                      This native build uses the CPU backend\n"
        "  --no-display                      Headless (always enabled)\n"
        "  --validate                        Exit 3 when validation rejects a result\n"
        "  --help                            Show this message\n";
}

int integer_value(const std::string& text, const std::string& option) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("Expected an integer after " + option);
    }
    return value;
}

double real_value(const std::string& text, const std::string& option) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("Expected a number after " + option);
    }
    return value;
}

Arguments parse_args(int argc, char** argv) {
    Arguments arguments;
    arguments.help = argc == 1;
    arguments.workflow.project_dir = ROBUST_ALIGNMENT_PROJECT_DIR;
    arguments.workflow.output_dir = "alignment-output-cpp";
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            arguments.help = true;
            continue;
        }
        if (option == "--demo-all") { arguments.demo_all = true; continue; }
        if (option == "--benchmark") { arguments.benchmark = true; continue; }
        if (option == "--synthetic-study") { arguments.synthetic = true; continue; }
        if (option == "--refine-ecc") { arguments.refine = true; continue; }
        if (option == "--dump-features") { arguments.dump_features = true; continue; }
        if (option == "--validate") { arguments.validate = true; continue; }
        if (option == "--no-display") { continue; }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + option);
        }
        const std::string value = argv[++index];
        if (option == "--reference") { arguments.reference = value; }
        else if (option == "--moving") { arguments.moving = value; }
        else if (option == "--output-dir") { arguments.workflow.output_dir = value; }
        else if (option == "--project-dir") { arguments.workflow.project_dir = value; }
        else if (option == "--assets-dir") { arguments.workflow.assets_dir = value; }
        else if (option == "--model-dir") { arguments.workflow.model_dir = value; }
        else if (option == "--correspondences") { arguments.correspondences = value; }
        else if (option == "--method") { arguments.method = value; }
        else if (option == "--estimator") { arguments.estimator = value; }
        else if (option == "--max-keypoints") {
            arguments.workflow.max_keypoints = integer_value(value, option);
        } else if (option == "--resize") { arguments.workflow.resize = integer_value(value, option); }
        else if (option == "--repeats") { arguments.workflow.repeats = integer_value(value, option); }
        else if (option == "--study-seeds") { arguments.workflow.study_seeds = integer_value(value, option); }
        else if (option == "--threshold") {
            arguments.workflow.config.reprojection_threshold = real_value(value, option);
        } else if (option == "--min-inliers") {
            arguments.workflow.config.min_inliers = integer_value(value, option);
        } else if (option == "--device") {
            if (value != "cpu") {
                throw std::invalid_argument("The native ONNX Runtime companion supports --device cpu.");
            }
        } else { throw std::invalid_argument("Unknown option: " + option); }
    }
    if (arguments.help) { return arguments; }
    const int modes = static_cast<int>(arguments.demo_all) + static_cast<int>(arguments.benchmark)
                    + static_cast<int>(arguments.synthetic);
    if (modes > 1 || (modes && (!arguments.reference.empty() || !arguments.moving.empty()))) {
        throw std::invalid_argument("Choose one workflow or supply a reference/moving image pair.");
    }
    if (!modes && (arguments.reference.empty() || arguments.moving.empty())) {
        throw std::invalid_argument("Supply both --reference and --moving, or choose --demo-all.");
    }
    if (arguments.method != "sift" && arguments.method != "aliked-nn" && arguments.method != "lightglue") {
        throw std::invalid_argument("method must be sift, aliked-nn, or lightglue.");
    }
    if (arguments.estimator != "ransac" && arguments.estimator != "magsac") {
        throw std::invalid_argument("estimator must be ransac or magsac.");
    }
    if (arguments.workflow.max_keypoints < 4 || arguments.workflow.resize < 0
        || (arguments.workflow.resize > 0 && arguments.workflow.resize < 16)
        || arguments.workflow.repeats < 1 || arguments.workflow.study_seeds < 1) {
        throw std::invalid_argument("Use >=4 keypoints, resize 0 or >=16, and positive repetition counts.");
    }
    arguments.workflow.config.validate();
    if (arguments.workflow.model_dir.empty()) {
        arguments.workflow.model_dir = arguments.workflow.project_dir / "cpp" / "models";
    }
    return arguments;
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write image: " + path.string());
    }
}

std::vector<cv::Point2f> read_points(const cv::FileNode& node) {
    std::vector<cv::Point2f> points;
    for (const auto& item : node) {
        std::vector<float> xy;
        item >> xy;
        if (xy.size() != 2) { throw std::invalid_argument("Each supplied point must contain x and y."); }
        points.emplace_back(xy[0], xy[1]);
    }
    return points;
}

alignment::MatchResult read_correspondences(const fs::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) { throw std::runtime_error("Cannot read correspondences: " + path.string()); }
    alignment::MatchResult matches;
    matches.reference_points = read_points(storage["reference_points"]);
    matches.moving_points = read_points(storage["moving_points"]);
    if (!storage["scores"].empty()) { storage["scores"] >> matches.scores; }
    matches.reference_keypoints = static_cast<int>(matches.reference_points.size());
    matches.moving_keypoints = static_cast<int>(matches.moving_points.size());
    matches.validate();
    return matches;
}

void dump_features(const fs::path& path, const alignment::FeatureSet& features) {
    cv::FileStorage storage(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) { throw std::runtime_error("Cannot write features: " + path.string()); }
    storage << "method" << features.method << "image_width" << features.image_size.width;
    storage << "image_height" << features.image_size.height << "keypoints" << "[";
    for (const auto& point : features.keypoints) { storage << "[" << point.x << point.y << "]"; }
    storage << "]" << "descriptors" << features.descriptors;
}

bool run_pair(const Arguments& arguments) {
    const cv::Mat reference = alignment::load_image(arguments.reference);
    const cv::Mat moving = alignment::load_image(arguments.moving);
    alignment::MatchResult matches;
    alignment::FeatureSet reference_features;
    alignment::FeatureSet moving_features;
    if (arguments.correspondences.empty()) {
        alignment::FeatureMatcher::Options options;
        options.method = arguments.method;
        options.model_dir = arguments.workflow.model_dir;
        options.max_keypoints = arguments.workflow.max_keypoints;
        options.resize = arguments.workflow.resize;
        alignment::FeatureMatcher matcher(options);
        reference_features = matcher.extract(reference);
        moving_features = matcher.extract(moving);
        matches = matcher.match(reference_features, moving_features);
    } else {
        matches = read_correspondences(arguments.correspondences);
    }
    const auto result = alignment::estimate_alignment(matches, reference.size(), moving.size(),
                                                      arguments.workflow.config, arguments.estimator);
    const fs::path output = arguments.workflow.output_dir;
    fs::create_directories(output);
    if (!arguments.dump_features || !arguments.correspondences.empty()) {
        for (const auto* name : {"reference_features.json", "moving_features.json", "correspondences.json"}) {
            fs::remove(output / name);
        }
    }
    write_image(output / "matches.jpg", alignment::draw_matches(reference, moving, matches, &result));
    // Clear only files owned by a previous refinement; its result is never silently reused.
    fs::remove(output / "refined_aligned.png");
    fs::remove(output / "refined_valid_mask.png");
    cv::FileStorage extra("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    extra << "opencv_version" << CV_VERSION << "method" << matches.method << "device" << "cpu";
    extra << "opencv_threads_requested" << 1 << "opencv_threads_reported" << cv::getNumThreads();
    extra << "onnxruntime_version" << alignment::native_inference_runtime_version();
    extra << "onnxruntime_intra_op_threads" << alignment::native_inference_intra_op_threads;
    extra << "onnxruntime_inter_op_threads" << alignment::native_inference_inter_op_threads;
    const std::string backend = !arguments.correspondences.empty() ? "provided_correspondences"
        : (arguments.method == "sift" ? "opencv_sift" : "onnxruntime_cpu");
    extra << "inference_backend" << backend;
    extra << "reference_keypoints" << matches.reference_keypoints << "moving_keypoints" << matches.moving_keypoints;
    extra << "extraction_ms" << matches.extraction_ms << "matching_ms" << matches.matching_ms;
    extra << "inliers" << "[";
    for (const auto flag : result.inliers) { extra << static_cast<int>(flag); }
    extra << "]";
    extra << "configuration" << "{";
    arguments.workflow.config.write(extra);
    extra << "}";
    if (result.accepted) {
        const auto warped = alignment::warp_alignment(moving, result, reference.size());
        cv::Mat overlay;
        cv::addWeighted(reference, 0.5, warped.first, 0.5, 0.0, overlay);
        reference.copyTo(overlay, warped.second == 0);
        cv::Mat outlined = moving.clone();
        std::vector<cv::Point> polygon;
        for (const auto& point : result.projected_reference_quad) {
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
        }
        cv::polylines(outlined, polygon, true, cv::Scalar(40, 220, 40), 3, cv::LINE_AA);
        write_image(output / "aligned.png", warped.first);
        write_image(output / "valid_mask.png", warped.second);
        write_image(output / "overlay.jpg", overlay);
        write_image(output / "reference_boundary.jpg", outlined);
        if (arguments.refine) {
            try {
                const auto refined = alignment::refine_ecc(reference, moving, result.H);
                alignment::AlignmentResult candidate = result;
                candidate.H = refined.first;
                const auto images = alignment::warp_alignment(moving, candidate, reference.size());
                write_image(output / "refined_aligned.png", images.first);
                write_image(output / "refined_valid_mask.png", images.second);
                extra << "ecc_candidate" << "{" << "correlation" << refined.second << "H" << refined.first << "}";
            } catch (const cv::Exception& error) {
                extra << "ecc_error" << error.what();
            } catch (const std::exception& error) {
                extra << "ecc_error" << error.what();
            }
        }
    } else {
        for (const auto* name : {"aligned.png", "valid_mask.png", "overlay.jpg", "reference_boundary.jpg"}) {
            fs::remove(output / name);
        }
    }
    if (arguments.dump_features && arguments.correspondences.empty()) {
        dump_features(output / "reference_features.json", reference_features);
        dump_features(output / "moving_features.json", moving_features);
        cv::FileStorage paired((output / "correspondences.json").string(),
                               cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!paired.isOpened()) { throw std::runtime_error("Cannot write correspondences.json"); }
        paired << "reference_points" << "[";
        for (const auto& point : matches.reference_points) { paired << "[" << point.x << point.y << "]"; }
        paired << "]" << "moving_points" << "[";
        for (const auto& point : matches.moving_points) { paired << "[" << point.x << point.y << "]"; }
        paired << "]" << "scores" << matches.scores;
    }
    // Merge the geometry serializer (true/false/null) with FileStorage's escaped metadata.
    std::string report = result.to_json();
    report.pop_back();
    const std::string metadata = extra.releaseAndGetString();
    const auto opening = metadata.find('{');
    report += ',' + metadata.substr(opening + 1);
    std::ofstream stream(output / "report.json");
    if (!stream || !(stream << report << '\n')) { throw std::runtime_error("Cannot write report.json"); }
    std::cout << report << '\n';
    return result.accepted;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_args(argc, argv);
        if (arguments.help || argc == 1) { show_help(); return 0; }
        cv::setNumThreads(1);  // Match the recorded geometry/threading protocol.
        bool accepted = false;
        if (arguments.demo_all) { accepted = alignment::run_demo_all(arguments.workflow); }
        else if (arguments.benchmark) { accepted = alignment::run_benchmark(arguments.workflow); }
        else if (arguments.synthetic) { accepted = alignment::run_synthetic_study(arguments.workflow); }
        else { accepted = run_pair(arguments); }
        if (arguments.validate) {
            std::cout << (accepted ? "VALIDATION PASSED\n" : "VALIDATION FAILED\n");
            return accepted ? 0 : 3;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
