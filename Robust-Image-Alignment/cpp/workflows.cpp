#include "workflows.hpp"
#include "feature_matcher.hpp"

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace robust_alignment {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct PairSpec {
    std::string id;
    fs::path reference;
    fs::path moving;
    std::string split;
    std::string change;
    cv::Mat truth;
};

struct CaseRun {
    cv::Mat reference;
    cv::Mat moving;
    MatchResult matches;
    AlignmentResult result;
    GeometricError error;
};

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate a percentile of an empty sample.");
    }
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    return values[low] + (position - static_cast<double>(low)) * (values[high] - values[low]);
}

fs::path assets_root(const WorkflowOptions& options) {
    return options.assets_dir.empty() ? options.project_dir / "assets" / "downloads" : options.assets_dir;
}

cv::FileStorage open_json(const fs::path& path) {
    fs::create_directories(path.parent_path());
    cv::FileStorage storage(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        throw std::runtime_error("Cannot write JSON report: " + path.string());
    }
    return storage;
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("Cannot write image: " + path.string());
    }
}

std::vector<PairSpec> read_pairs(const WorkflowOptions& options) {
    const fs::path path = options.project_dir / "assets_manifest.json";
    cv::FileStorage storage(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        throw std::runtime_error("Cannot read sample manifest: " + path.string());
    }
    std::vector<PairSpec> pairs;
    for (const auto& node : storage["pairs"]) {
        PairSpec pair;
        node["id"] >> pair.id;
        std::string reference;
        std::string moving;
        node["reference"] >> reference;
        node["moving"] >> moving;
        pair.reference = assets_root(options) / reference;
        pair.moving = assets_root(options) / moving;
        node["split"] >> pair.split;
        node["change"] >> pair.change;
        std::vector<double> values;
        node["homography_matrix"] >> values;
        if (values.size() != 9) {
            throw std::runtime_error("Expected nine ground-truth values for " + pair.id);
        }
        pair.truth = cv::Mat(3, 3, CV_64F);
        std::copy(values.begin(), values.end(), pair.truth.ptr<double>());
        pairs.push_back(std::move(pair));
    }
    if (pairs.size() != 10) {
        throw std::runtime_error("The companion manifest must contain ten real-image pairs.");
    }
    return pairs;
}

FeatureMatcher::Options matcher_options(const WorkflowOptions& options, const std::string& method) {
    FeatureMatcher::Options result;
    result.method = method;
    result.model_dir = options.model_dir;
    result.max_keypoints = options.max_keypoints;
    result.resize = options.resize;
    result.ratio = 0.8F;
    return result;
}

void write_case_images(const fs::path& directory, const CaseRun& run, bool diagnostic = false) {
    fs::create_directories(directory);
    write_image(directory / "reference.png", run.reference);
    write_image(directory / "moving.png", run.moving);
    write_image(directory / "matches.jpg", draw_matches(run.reference, run.moving, run.matches, &run.result));
    if (!run.result.H.empty() && (run.result.accepted || diagnostic)) {
        const auto warped = warp_alignment(run.moving, run.result, run.reference.size(), !diagnostic);
        cv::Mat overlay;
        cv::addWeighted(run.reference, 0.5, warped.first, 0.5, 0.0, overlay);
        run.reference.copyTo(overlay, warped.second == 0);
        write_image(directory / "aligned.png", warped.first);
        write_image(directory / "valid_mask.png", warped.second);
        write_image(directory / "overlay.jpg", overlay);
        if (run.result.accepted) {
            cv::Mat outlined = run.moving.clone();
            std::vector<cv::Point> polygon;
            for (const auto& point : run.result.projected_reference_quad) {
                polygon.emplace_back(cvRound(point.x), cvRound(point.y));
            }
            cv::polylines(outlined, std::vector<std::vector<cv::Point>>{polygon}, true,
                          cv::Scalar(40, 220, 40), 3, cv::LINE_AA);
            write_image(directory / "reference_boundary.jpg", outlined);
        }
    } else {
        // Remove only files owned by a previous successful run in this directory.
        for (const auto* name : {"aligned.png", "valid_mask.png", "overlay.jpg", "reference_boundary.jpg"}) {
            fs::remove(directory / name);
        }
    }
    auto report = open_json(directory / "report.json");
    report << "opencv_version" << CV_VERSION << "result" << "{";
    run.result.write(report);
    report << "}" << "independent_error" << "{";
    run.error.write(report);
    report << "}";
}

CaseRun match_case(FeatureMatcher& matcher, const fs::path& reference, const fs::path& moving,
                   const WorkflowOptions& options, const cv::Mat& truth = cv::Mat()) {
    CaseRun run;
    run.reference = load_image(reference);
    run.moving = load_image(moving);
    run.matches = matcher.match_images(run.reference, run.moving);
    run.result = estimate_alignment(run.matches, run.reference.size(), run.moving.size(), options.config, "magsac");
    if (!truth.empty()) {
        run.error = geometric_error(run.result.H, truth, run.reference.size(), run.moving.size());
    }
    return run;
}

void write_optional(cv::FileStorage& storage, const char* name, const std::optional<double>& value) {
    if (value) {
        storage << name << *value;
    }
}

void write_configuration(cv::FileStorage& storage, const WorkflowOptions& options) {
    storage << "opencv_version" << CV_VERSION;
    storage << "runtime" << "native_cpp";
    storage << "onnxruntime_version" << native_inference_runtime_version();
    storage << "neural_device" << "cpu";
    // FileStorage limits individual JSON strings, so retain the full build in a sidecar.
    const fs::path build_path = options.output_dir / "opencv_build.txt";
    std::ofstream build_information(build_path);
    if (!build_information) {
        throw std::runtime_error("Cannot write OpenCV build information: " + build_path.string());
    }
    build_information << cv::getBuildInformation();
    if (!build_information) {
        throw std::runtime_error("Failed while writing OpenCV build information.");
    }
    storage << "opencv_build_information_file" << "opencv_build.txt";
#if defined(__clang__)
    storage << "compiler" << std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    storage << "compiler" << std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    storage << "compiler" << std::string("MSVC ") + std::to_string(_MSC_VER);
#else
    storage << "compiler" << "not identified";
#endif
    storage << "configuration" << "{";
    storage << "max_keypoints" << options.max_keypoints << "resize_longest_side_cap" << options.resize;
    storage << "repeats" << options.repeats << "study_seeds" << options.study_seeds;
    storage << "threshold_px" << options.config.reprojection_threshold;
    // GCD reports its pool capacity even when setNumThreads(1) bypasses parallel_for_.
    storage << "opencv_threads_requested" << 1 << "opencv_threads_reported" << cv::getNumThreads();
    storage << "onnxruntime_intra_op_threads" << native_inference_intra_op_threads;
    storage << "onnxruntime_inter_op_threads" << native_inference_inter_op_threads;
    storage << "homography_direction" << "moving_to_reference";
    storage << "timing_excludes" << "[" << "disk_io" << "model_load" << "warm_up" << "visualization" << "]";
    storage << "}";
}

void write_ecc(cv::FileStorage& storage, const std::string& id, const CaseRun& run,
               const cv::Mat& truth, const fs::path& output_dir) {
    storage << "{" << "id" << id << "before" << "{";
    geometric_error(run.result.H, truth, run.reference.size(), run.moving.size()).write(storage);
    storage << "}";
    if (run.result.H.empty()) {
        storage << "status" << "no_initial_model";
    } else {
        try {
            const auto start = Clock::now();
            const auto refined = refine_ecc(run.reference, run.moving, run.result.H);
            storage << "status" << "returned" << "correlation" << refined.second;
            storage << "refinement_ms" << elapsed_ms(start) << "after" << "{";
            geometric_error(refined.first, truth, run.reference.size(), run.moving.size()).write(storage);
            storage << "}";
            AlignmentResult diagnostic = run.result;
            diagnostic.H = refined.first;
            const auto warped = warp_alignment(run.moving, diagnostic, run.reference.size(), false);
            write_image(output_dir / (id + "_ecc.png"), warped.first);
        } catch (const std::exception& error) {
            storage << "status" << "failed" << "error" << error.what();
        }
    }
    storage << "}";
}

int write_synthetic_rows(cv::FileStorage& storage, const WorkflowOptions& options) {
    if (options.study_seeds < 1) {
        throw std::invalid_argument("study_seeds must be positive.");
    }
    const cv::Mat truth(cv::Matx33d(
        0.87, -0.15, 65.0, 0.08, 0.92, 40.0, 0.00012, -0.00009, 1.0));
    int rows = 0;
    for (const double noise : {0.5, 1.5, 3.0}) {
        for (int seed = 0; seed < options.study_seeds; ++seed) {
            // A named native generator makes this experiment reproducible without NumPy.
            std::mt19937 random(static_cast<std::mt19937::result_type>(seed + 1000));
            std::uniform_real_distribution<float> x(50.0F, 750.0F);
            std::uniform_real_distribution<float> y(50.0F, 550.0F);
            std::normal_distribution<float> jitter(0.0F, static_cast<float>(noise));
            MatchResult matches;
            for (int index = 0; index < 160; ++index) {
                // Name the draws so argument-evaluation order cannot swap the random stream.
                const float point_x = x(random);
                const float point_y = y(random);
                matches.reference_points.emplace_back(point_x, point_y);
            }
            matches.moving_points = project_points(matches.reference_points, truth);
            for (auto& point : matches.moving_points) {
                point.x += jitter(random);
                point.y += jitter(random);
            }
            std::vector<int> indices(160);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), random);
            std::uniform_real_distribution<float> outlier_x(0.0F, 800.0F);
            std::uniform_real_distribution<float> outlier_y(0.0F, 600.0F);
            for (int index = 0; index < 96; ++index) {
                const float point_x = outlier_x(random);
                const float point_y = outlier_y(random);
                matches.moving_points[static_cast<std::size_t>(indices[static_cast<std::size_t>(index)])] =
                    cv::Point2f(point_x, point_y);
            }
            for (const double threshold : {1.0, 3.0, 5.0}) {
                AlignmentConfig config = options.config;
                config.reprojection_threshold = threshold;
                config.max_median_reprojection_error = threshold;
                for (const auto* estimator : {"ransac", "magsac"}) {
                    const auto result = estimate_alignment(matches, cv::Size(800, 600), cv::Size(800, 600), config, estimator);
                    storage << "{" << "noise_sigma_px" << noise << "outlier_fraction" << 0.6;
                    storage << "seed" << seed << "threshold_px" << threshold;
                    result.write(storage);
                    geometric_error(result.H, truth, cv::Size(800, 600), cv::Size(800, 600)).write(storage);
                    storage << "}";
                    ++rows;
                }
            }
        }
    }
    return rows;
}

}  // namespace

void GeometricError::write(cv::FileStorage& storage) const {
    // Qualitative document/object cases have an estimate but no independent GT error.
    storage << "available" << static_cast<int>(has_model);
    write_optional(storage, "grid_median_error_px", grid_median_error_px);
    write_optional(storage, "grid_p95_error_px", grid_p95_error_px);
    write_optional(storage, "corner_mean_error_px", corner_mean_error_px);
    storage << "grid_points" << grid_points;
}

GeometricError geometric_error(const cv::Mat& moving_to_reference, const cv::Mat& truth_reference_to_moving,
                               cv::Size reference_size, cv::Size moving_size) {
    GeometricError error;
    if (moving_to_reference.empty()) {
        return error;
    }
    // Use exactly the notebook's 30-by-24 grid and eight-pixel overlap margin.
    std::vector<cv::Point2f> reference;
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 30; ++x) {
            reference.emplace_back(8.0F + static_cast<float>(x) * static_cast<float>(reference_size.width - 17) / 29.0F,
                                   8.0F + static_cast<float>(y) * static_cast<float>(reference_size.height - 17) / 23.0F);
        }
    }
    const auto moving = project_points(reference, truth_reference_to_moving);
    std::vector<cv::Point2f> overlap_reference;
    std::vector<cv::Point2f> overlap_moving;
    for (std::size_t index = 0; index < moving.size(); ++index) {
        const auto& point = moving[index];
        if (point.x >= 8.0F && point.y >= 8.0F && point.x < static_cast<float>(moving_size.width - 8)
            && point.y < static_cast<float>(moving_size.height - 8)) {
            overlap_reference.push_back(reference[index]);
            overlap_moving.push_back(point);
        }
    }
    if (overlap_moving.empty()) {
        throw std::runtime_error("Ground-truth transformation has no sampled overlap.");
    }
    const auto recovered = project_points(overlap_moving, moving_to_reference);
    std::vector<double> distances;
    for (std::size_t index = 0; index < recovered.size(); ++index) {
        distances.push_back(cv::norm(recovered[index] - overlap_reference[index]));
    }
    error.has_model = true;
    error.grid_points = static_cast<int>(distances.size());
    error.grid_median_error_px = percentile(distances, 0.5);
    error.grid_p95_error_px = percentile(distances, 0.95);
    const auto corners = image_corners(reference_size);
    const auto recovered_corners = project_points(project_points(corners, truth_reference_to_moving), moving_to_reference);
    double corner_sum = 0.0;
    for (std::size_t index = 0; index < corners.size(); ++index) {
        corner_sum += cv::norm(recovered_corners[index] - corners[index]);
    }
    error.corner_mean_error_px = corner_sum / static_cast<double>(corners.size());
    return error;
}

bool run_synthetic_study(const WorkflowOptions& options) {
    auto storage = open_json(options.output_dir / "synthetic_study.json");
    write_configuration(storage, options);
    storage << "capture_type" << "synthetic_correspondences";
    storage << "random_generator" << "std::mt19937 with native standard-library distributions; not NumPy PCG64";
    storage << "rows" << "[";
    const int count = write_synthetic_rows(storage, options);
    storage << "]" << "row_count" << count;
    std::cout << "Synthetic estimator study: " << count << " rows\n";
    return count == options.study_seeds * 18;
}

bool run_demo_all(const WorkflowOptions& options) {
    const auto pairs = read_pairs(options);
    FeatureMatcher matcher(matcher_options(options, "lightglue"));
    std::map<std::string, PairSpec> by_id;
    for (const auto& pair : pairs) {
        by_id.emplace(pair.id, pair);
    }
    const auto assets = assets_root(options);
    fs::create_directories(options.output_dir);
    auto report = open_json(options.output_dir / "demo_report.json");
    write_configuration(report, options);
    report << "demonstrations" << "[";
    bool passed = true;
    std::map<std::string, CaseRun> cases;
    for (const auto* id : {"graf_1_6", "leuven_1_6"}) {
        const auto& pair = by_id.at(id);
        auto run = match_case(matcher, pair.reference, pair.moving, options, pair.truth);
        write_case_images(options.output_dir / id, run);
        // The notebook asserts acceptance here and reports independent accuracy separately.
        // CPU and GPU inference can change matching decisions, so accepted geometry can
        // miss the benchmark's stricter accuracy criterion on one computation backend.
        const bool correct = run.result.accepted;
        const bool benchmark_success = run.result.accepted && run.error.grid_median_error_px &&
                                       *run.error.grid_median_error_px < 3.0;
        passed = passed && correct;
        report << "{" << "id" << id << "expected_accepted" << 1 << "validation_passed" << static_cast<int>(correct);
        report << "benchmark_success_3px" << static_cast<int>(benchmark_success);
        run.result.write(report);
        run.error.write(report);
        report << "}";
        // Expose an accuracy miss even when the notebook's acceptance check passes.
        if (!benchmark_success) {
            std::cout << id << ": accepted=" << (run.result.accepted ? "true" : "false")
                      << ", independent median grid error=";
            if (run.error.grid_median_error_px) {
                std::cout << *run.error.grid_median_error_px << " px";
            } else {
                std::cout << "unavailable";
            }
            std::cout << "; benchmark_success_3px=false (requires accepted geometry and error < 3 px)."
                      << std::endl;
        }
        cases.emplace(id, std::move(run));
    }
    auto document = match_case(matcher, assets / "learnopencv/form.jpg", assets / "learnopencv/scanned-form.jpg", options);
    write_case_images(options.output_dir / "document", document);
    passed = passed && document.result.accepted;
    report << "{" << "id" << "document" << "expected_accepted" << 1;
    document.result.write(report);
    report << "}";
    cases.emplace("document", std::move(document));

    // Cache the single reference and scene feature sets for identification and catalogue reuse.
    const cv::Mat target = load_image(assets / "opencv/box.png");
    const cv::Mat scene = load_image(assets / "opencv/box_in_scene.png");
    const auto enrolled = matcher.extract(target);
    const auto query = matcher.extract(scene);
    CaseRun identified;
    identified.reference = target;
    identified.moving = scene;
    identified.matches = matcher.match(enrolled, query);
    identified.result = estimate_alignment(identified.matches, target.size(), scene.size(), options.config);
    write_case_images(options.output_dir / "box_identification", identified);
    passed = passed && identified.result.accepted;
    report << "{" << "id" << "box_identification" << "expected_accepted" << 1;
    identified.result.write(report);
    report << "}";
    for (const auto* file : {"fruits.jpg", "building.jpg"}) {
        CaseRun negative;
        negative.reference = target;
        negative.moving = load_image(assets / "opencv" / file);
        negative.matches = matcher.match(enrolled, matcher.extract(negative.moving));
        negative.result = estimate_alignment(negative.matches, target.size(), negative.moving.size(), options.config);
        const std::string id = std::string("absent_") + fs::path(file).stem().string();
        write_case_images(options.output_dir / id, negative);
        passed = passed && !negative.result.accepted;
        report << "{" << "id" << id << "expected_accepted" << 0;
        negative.result.write(report);
        report << "}";
    }
    auto aloe = match_case(matcher, assets / "opencv/aloeL.jpg", assets / "opencv/aloeR.jpg", options);
    write_case_images(options.output_dir / "aloe_parallax", aloe, true);
    report << "{" << "id" << "aloe_parallax" << "purpose" << "diagnostic: acceptance does not imply all scene depths align";
    aloe.result.write(report);
    report << "}" << "]";

    report << "catalogue" << "[";
    std::vector<std::string> accepted_names;
    const auto graffiti_features = matcher.extract(cases.at("graf_1_6").reference);
    const auto form_features = matcher.extract(cases.at("document").reference);
    for (const auto& entry : std::array<std::pair<std::string, FeatureSet>, 3>{{
             {"box design", enrolled}, {"graffiti wall", graffiti_features}, {"blank form", form_features}}}) {
        const auto found = matcher.match(entry.second, query);
        const auto result = estimate_alignment(found, entry.second.image_size, scene.size(), options.config);
        report << "{" << "name" << entry.first;
        result.write(report);
        report << "}";
        if (result.accepted) {
            accepted_names.push_back(entry.first);
        }
    }
    report << "]" << "catalogue_identification";
    report << (accepted_names.size() == 1 ? accepted_names.front() : "ambiguous or absent");
    passed = passed && accepted_names.size() == 1 && accepted_names.front() == "box design";

    const auto& graffiti = cases.at("graf_1_6");
    report << "fixed_match_estimator_comparison" << "[";
    for (const auto* estimator : {"ransac", "magsac"}) {
        const auto result = estimate_alignment(graffiti.matches, graffiti.reference.size(), graffiti.moving.size(), options.config, estimator);
        report << "{";
        result.write(report);
        geometric_error(result.H, by_id.at("graf_1_6").truth, graffiti.reference.size(), graffiti.moving.size()).write(report);
        report << "}";
    }
    report << "]" << "ecc_comparison" << "[";
    for (const auto* id : {"graf_1_6", "leuven_1_6"}) {
        write_ecc(report, id, cases.at(id), by_id.at(id).truth, options.output_dir);
    }
    report << "]";

    // Preserve only a small spatial patch to reproduce the notebook's rejection exercise.
    MatchResult patch;
    for (std::size_t index = 0; index < graffiti.matches.reference_points.size(); ++index) {
        const auto& point = graffiti.matches.reference_points[index];
        if (point.x < static_cast<float>(graffiti.reference.cols) / 4.0F
            && point.y < static_cast<float>(graffiti.reference.rows) / 4.0F) {
            patch.reference_points.push_back(point);
            patch.moving_points.push_back(graffiti.matches.moving_points[index]);
        }
    }
    AlignmentConfig strict = options.config;
    strict.min_reference_coverage = 0.10;
    const auto patch_result = estimate_alignment(patch, graffiti.reference.size(), graffiti.moving.size(), strict);
    passed = passed && !patch_result.accepted;
    report << "spatial_support_exercise" << "{";
    patch_result.write(report);
    report << "}" << "validation_passed" << static_cast<int>(passed);
    report.release();
    passed = run_synthetic_study(options) && passed;
    std::cout << "Demo validation " << (passed ? "passed" : "failed") << ": " << options.output_dir << '\n';
    return passed;
}

bool run_benchmark(const WorkflowOptions& options) {
    if (options.repeats < 1) {
        throw std::invalid_argument("repeats must be positive.");
    }
    const auto pairs = read_pairs(options);
    FeatureMatcher neural(matcher_options(options, "lightglue"));
    FeatureMatcher nearest(matcher_options(options, "aliked-nn"));
    FeatureMatcher sift(matcher_options(options, "sift"));
    auto report = open_json(options.output_dir / "benchmark_report.json");
    write_configuration(report, options);
    report << "protocol" << "two validation pairs; eight evaluation pairs from two capture sequences";
    report << "success_criterion" << "accepted and median independent overlap-grid error below three original reference pixels";
    report << "real_pairs" << "[";
    std::ofstream csv(options.output_dir / "benchmark.csv");
    if (!csv) {
        throw std::runtime_error("Cannot write benchmark CSV.");
    }
    csv << "id,split,repeat,method,estimator,accepted,success_3px,grid_median_error_px,grid_p95_error_px,extraction_ms,matching_ms,geometry_stage_ms,warping_ms,pipeline_ms\n";
    int rows = 0;
    std::map<std::string, std::vector<bool>> successes;
    std::map<std::string, std::vector<std::pair<std::string, double>>> timing;
    for (const auto& pair : pairs) {
        const cv::Mat reference = load_image(pair.reference);
        const cv::Mat moving = load_image(pair.moving);
        const auto warm_reference = neural.extract(reference);
        const auto warm_moving = neural.extract(moving);
        const auto warm_matches = neural.match(warm_reference, warm_moving);
        nearest.match(warm_reference, warm_moving);
        sift.match_images(reference, moving);
        for (const auto* estimator : {"ransac", "magsac"}) {
            const auto result = estimate_alignment(warm_matches, reference.size(), moving.size(), options.config, estimator);
            if (result.accepted) {
                warp_alignment(moving, result, reference.size());
            }
        }
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            auto start = Clock::now();
            const auto reference_neural = neural.extract(reference);
            const auto moving_neural = neural.extract(moving);
            const double neural_extraction_ms = elapsed_ms(start);
            start = Clock::now();
            const auto reference_sift = sift.extract(reference);
            const auto moving_sift = sift.extract(moving);
            const double sift_extraction_ms = elapsed_ms(start);
            struct MethodRun { std::string name; MatchResult matches; double extraction_ms; double matching_ms; };
            std::vector<MethodRun> methods;
            for (const auto* method : {"sift", "aliked-nn", "lightglue"}) {
                start = Clock::now();
                MatchResult matches;
                if (std::string(method) == "sift") {
                    matches = sift.match(reference_sift, moving_sift);
                } else if (std::string(method) == "aliked-nn") {
                    matches = nearest.match(reference_neural, moving_neural);
                } else {
                    matches = neural.match(reference_neural, moving_neural);
                }
                methods.push_back({method, std::move(matches), std::string(method) == "sift" ? sift_extraction_ms : neural_extraction_ms, elapsed_ms(start)});
            }
            for (const auto& method : methods) {
                for (const auto* estimator : {"ransac", "magsac"}) {
                    start = Clock::now();
                    const auto result = estimate_alignment(method.matches, reference.size(), moving.size(), options.config, estimator);
                    const double geometry_ms = elapsed_ms(start);
                    double warping_ms = 0.0;
                    if (result.accepted) {
                        start = Clock::now();
                        warp_alignment(moving, result, reference.size());
                        warping_ms = elapsed_ms(start);
                    }
                    const auto error = geometric_error(result.H, pair.truth, reference.size(), moving.size());
                    const bool success = result.accepted && error.grid_median_error_px && *error.grid_median_error_px < 3.0;
                    const double pipeline_ms = method.extraction_ms + method.matching_ms + geometry_ms + warping_ms;
                    report << "{" << "id" << pair.id << "split" << pair.split << "change" << pair.change;
                    report << "repeat" << repeat << "method" << method.name;
                    result.write(report);
                    error.write(report);
                    report << "success_3px" << static_cast<int>(success);
                    report << "extraction_ms" << method.extraction_ms << "matching_ms" << method.matching_ms;
                    report << "geometry_stage_ms" << geometry_ms << "warping_ms" << warping_ms << "pipeline_ms" << pipeline_ms << "}";
                    csv << pair.id << ',' << pair.split << ',' << repeat << ',' << method.name << ',' << estimator << ',' << result.accepted << ',' << success << ',';
                    if (error.grid_median_error_px) { csv << *error.grid_median_error_px; }
                    csv << ',';
                    if (error.grid_p95_error_px) { csv << *error.grid_p95_error_px; }
                    csv << ',' << method.extraction_ms << ',' << method.matching_ms << ',' << geometry_ms << ',' << warping_ms << ',' << pipeline_ms << '\n';
                    if (pair.split == "evaluation" && std::string(estimator) == "magsac") {
                        if (repeat == 0) {
                            successes[pair.id].push_back(success);
                        }
                        timing[method.name].emplace_back(pair.id, pipeline_ms);
                    }
                    ++rows;
                }
            }
        }
        std::cout << "Completed native benchmark pair " << pair.id << std::endl;
    }
    report << "]" << "row_count" << rows << "common_success_timing_summary" << "[";
    for (const auto* method : {"sift", "aliked-nn", "lightglue"}) {
        std::vector<double> values;
        for (const auto& measurement : timing[method]) {
            const auto& outcomes = successes.at(measurement.first);
            if (outcomes.size() == 3 && std::all_of(outcomes.begin(), outcomes.end(), [](bool value) { return value; })) {
                values.push_back(measurement.second);
            }
        }
        report << "{" << "method" << method << "timing_trials" << static_cast<int>(values.size());
        if (!values.empty()) {
            report << "p10_ms" << percentile(values, 0.1) << "median_ms" << percentile(values, 0.5) << "p90_ms" << percentile(values, 0.9);
        }
        report << "}";
    }
    report << "]" << "synthetic_study" << "[";
    const int synthetic_rows = write_synthetic_rows(report, options);
    report << "]" << "synthetic_row_count" << synthetic_rows;
    return rows == static_cast<int>(pairs.size()) * options.repeats * 6 && synthetic_rows == options.study_seeds * 18;
}

}  // namespace robust_alignment
