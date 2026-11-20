// ALIKED algorithms: copyright (c) 2022 Zhao Xiaoming, BSD-3-Clause.
// LightGlue algorithms: cvg/LightGlue contributors, Apache-2.0.
// Native adaptations retain the complete source notices in models/*-LICENSE.
#include "feature_matcher.hpp"

#include <opencv2/features.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace robust_alignment {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start) {
    // A monotonic clock keeps duration measurements independent of clock resets.
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check_image(const cv::Mat& image) {
    // imread supplies this contract; reject incompatible caller-owned matrices.
    if (image.type() != CV_8UC3 || image.cols < 2 || image.rows < 2) {
        throw std::invalid_argument("Expected a uint8 BGR image at least 2 by 2 pixels");
    }
}

cv::Mat resize_rgb_float(const cv::Mat& image_bgr, cv::Size size) {
    // LightGlue's Kornia preprocessor converts RGB to float before resizing.
    cv::Mat rgb, source;
    cv::cvtColor(image_bgr, rgb, cv::COLOR_BGR2RGB);
    // Learning models consume normalized intensities, not uint8 image values.
    rgb.convertTo(source, CV_32FC3, 1.0 / 255.0);
    if (size == image_bgr.size()) {
        return source;
    }
    // Compute the two actual scales after integer output dimensions are known.
    const double factor_y = static_cast<double>(source.rows) / size.height;
    const double factor_x = static_cast<double>(source.cols) / size.width;
    // Kornia antialiasing applies a Gaussian first, then ordinary half-pixel
    // bilinear interpolation. It is different from PyTorch's antialias flag.
    const double sigma_y = std::max((factor_y - 1.0) / 2.0, 0.001);
    const double sigma_x = std::max((factor_x - 1.0) / 2.0, 0.001);
    const auto gaussian = [](double sigma) {
        // Kornia chooses an odd support of approximately four sigma, at least 3.
        int size = static_cast<int>(std::max(4.0 * sigma, 3.0));
        size += (size % 2 == 0);
        cv::Mat kernel(1, size, CV_32F);
        float sum = 0;
        for (int i = 0; i < size; ++i) {
            // Center the 1-D kernel and evaluate its unnormalized Gaussian.
            const float distance = static_cast<float>(i - size / 2);
            const float value = std::exp(-distance * distance / static_cast<float>(2 * sigma * sigma));
            kernel.at<float>(i) = value;
            sum += value;
        }
        kernel /= sum;  // Preserve a constant image's intensity through filtering.
        return kernel;
    };
    cv::Mat filtered;
    // Reflection without repeating the edge pixel matches Kornia's reflect pad.
    cv::sepFilter2D(source, filtered, -1, gaussian(sigma_x), gaussian(sigma_y),
                    cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
    cv::Mat result(size, CV_32FC3);
    for (int y = 0; y < size.height; ++y) {
        // align_corners=False locates samples by their pixel centers.
        const double sy = std::max(0.0, (y + 0.5) * factor_y - 0.5);
        const int y0 = static_cast<int>(sy), y1 = std::min(y0 + 1, filtered.rows - 1);
        const float fy = static_cast<float>(sy - y0);
        for (int x = 0; x < size.width; ++x) {
            // Clamp the second neighbor at the image edge, as interpolate does.
            const double sx = std::max(0.0, (x + 0.5) * factor_x - 0.5);
            const int x0 = static_cast<int>(sx), x1 = std::min(x0 + 1, filtered.cols - 1);
            const float fx = static_cast<float>(sx - x0);
            // Blend four float RGB samples without uint8 rounding between steps.
            result.at<cv::Vec3f>(y, x) =
                (filtered.at<cv::Vec3f>(y0, x0) * (1 - fx) + filtered.at<cv::Vec3f>(y0, x1) * fx) * (1 - fy)
                + (filtered.at<cv::Vec3f>(y1, x0) * (1 - fx) + filtered.at<cv::Vec3f>(y1, x1) * fx) * fy;
        }
    }
    return result;
}

std::vector<cv::Point2f> detect_aliked(const cv::Mat& scores, int maximum) {
    // This is ALIKED DKD: two-round NMS, the original threshold fallback,
    // strongest-response cap, and a 5x5 soft-argmax subpixel refinement.
    const cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
    cv::Mat local_maximum, maxima;
    // A dilation is a local maximum filter; equality retains candidate peaks.
    cv::dilate(scores, local_maximum, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::compare(scores, local_maximum, maxima, cv::CMP_EQ);
    for (int round = 0; round < 2; ++round) {
        cv::Mat suppression, remaining = scores.clone(), new_maximum, new_mask, not_suppressed;
        // Existing maxima suppress their radius-2 neighborhood.
        cv::dilate(maxima, suppression, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        remaining.setTo(0, suppression);
        // A second pass recovers unsuppressed maxima hidden by that neighborhood.
        cv::dilate(remaining, new_maximum, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::compare(remaining, new_maximum, new_mask, cv::CMP_EQ);
        cv::bitwise_not(suppression, not_suppressed);
        cv::bitwise_and(new_mask, not_suppressed, new_mask);
        cv::bitwise_or(maxima, new_mask, maxima);
    }
    // Soft argmax needs the full 5x5 patch, so ignore the radius-2 image border.
    maxima.rowRange(0, 2).setTo(0);
    maxima.rowRange(scores.rows - 2, scores.rows).setTo(0);
    maxima.colRange(0, 2).setTo(0);
    maxima.colRange(scores.cols - 2, scores.cols).setTo(0);
    auto select = [&](float threshold) {
        std::vector<int> indices;
        for (int y = 2; y < scores.rows - 2; ++y) {
            for (int x = 2; x < scores.cols - 2; ++x) {
                // Row-major order agrees with upstream nonzero() ordering.
                if (maxima.at<std::uint8_t>(y, x) && scores.at<float>(y, x) > threshold) {
                    indices.push_back(y * scores.cols + x);
                }
            }
        }
        return indices;
    };
    auto indices = select(0.2F);
    if (indices.empty()) {
        // Upstream falls back to the mean dense score only when no peak passes.
        indices = select(static_cast<float>(cv::mean(scores)[0]));
    }
    if (indices.size() > static_cast<std::size_t>(maximum)) {
        // The budget selects strongest peaks; descriptor matching uses this order.
        std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
            return scores.at<float>(a / scores.cols, a % scores.cols) > scores.at<float>(b / scores.cols, b % scores.cols);
        });
        indices.resize(static_cast<std::size_t>(maximum));
    }
    std::vector<cv::Point2f> points;
    points.reserve(indices.size());
    for (int index : indices) {
        // Translate a flattened row-major score index into a pixel coordinate.
        const int x = index % scores.cols;
        const int y = index / scores.cols;
        float largest = -std::numeric_limits<float>::infinity();
        // Subtracting the patch maximum keeps exponentials numerically stable.
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                largest = std::max(largest, scores.at<float>(y + dy, x + dx));
            }
        }
        float denominator = 0.0F, residual_x = 0.0F, residual_y = 0.0F;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                const float weight = std::exp((scores.at<float>(y + dy, x + dx) - largest) / 0.1F);
                // A temperature of 0.1 is the original trained detector setting.
                denominator += weight;
                residual_x += dx * weight;
                residual_y += dy * weight;
            }
        }
        // The weighted patch offset refines an integer peak to subpixel precision.
        points.emplace_back(x + residual_x / denominator, y + residual_y / denominator);
    }
    return points;
}

std::vector<float> gather_rows(const std::vector<float>& source, int width, const std::vector<int>& rows) {
    // Pruning must gather descriptors and coordinates with identical row indices.
    std::vector<float> result(rows.size() * static_cast<std::size_t>(width));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::copy_n(source.data() + static_cast<std::size_t>(rows[i]) * width, width,
                    result.data() + i * static_cast<std::size_t>(width));
    }
    return result;
}

std::vector<float> copy_tensor(const Ort::Value& tensor) {
    // Copy outputs before the next session call releases their ORT-owned storage.
    const auto size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto* data = tensor.GetTensorData<float>();
    return std::vector<float>(data, data + size);
}

}  // namespace

std::string native_inference_runtime_version() {
    return OrtGetApiBase()->GetVersionString();
}

struct FeatureMatcher::Impl {
    explicit Impl(Options value) : options(std::move(value)) {
        // Keep method names identical to the notebook and Python command line.
        if (options.method != "sift" && options.method != "aliked-nn" && options.method != "lightglue") {
            throw std::invalid_argument("method must be sift, aliked-nn, or lightglue");
        }
        if (options.max_keypoints < 4 || options.resize < 0 || (options.resize > 0 && options.resize < 16)
            || !(options.ratio > 0.0F && options.ratio < 1.0F)) {
            throw std::invalid_argument("Invalid feature budget, resize cap, or ratio threshold");
        }
        if (options.method == "sift") {
            sift = cv::SIFT::create(options.max_keypoints);
        } else {
            // SIFT needs no ORT environment, sessions, model files, or thread pools.
            environment = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LearnOpenCV native alignment");
            session_options = std::make_unique<Ort::SessionOptions>();
            // A fixed CPU thread budget makes this portable path reproducible.
            session_options->SetIntraOpNumThreads(native_inference_intra_op_threads);
            session_options->SetInterOpNumThreads(native_inference_inter_op_threads);
            session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            environment->DisableTelemetryEvents();  // Models run entirely locally.
            // Load only the graphs needed by the selected matching route.
            dense = session("aliked_n16_dense.onnx");
            descriptor = session("aliked_n16_descriptor.onnx");
            if (options.method == "lightglue") {
                for (int layer = 0; layer < 9; ++layer) {
                    glue.push_back(session("lightglue_layer_" + std::to_string(layer) + ".onnx"));
                }
            }
        }
    }

    std::unique_ptr<Ort::Session> session(const std::string& name) {
        // Resolve models from the explicit directory, independent of the caller's cwd.
        const auto path = options.model_dir / name;
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("Missing native model: " + path.string());
        }
        // ORT owns weights and execution buffers for the lifetime of this session.
        return std::make_unique<Ort::Session>(*environment, path.c_str(), *session_options);
    }

    Options options;
    std::unique_ptr<Ort::Env> environment;
    std::unique_ptr<Ort::SessionOptions> session_options;
    cv::Ptr<cv::SIFT> sift;
    std::unique_ptr<Ort::Session> dense;
    std::unique_ptr<Ort::Session> descriptor;
    std::vector<std::unique_ptr<Ort::Session>> glue;
};

FeatureMatcher::FeatureMatcher(Options options) : impl_(std::make_unique<Impl>(std::move(options))) {}
FeatureMatcher::~FeatureMatcher() = default;
FeatureMatcher::FeatureMatcher(FeatureMatcher&&) noexcept = default;
FeatureMatcher& FeatureMatcher::operator=(FeatureMatcher&&) noexcept = default;

FeatureSet FeatureMatcher::extract(const cv::Mat& image_bgr) {
    check_image(image_bgr);
    const auto started = Clock::now();
    FeatureSet result;
    result.image_size = image_bgr.size();  // Matching geometry uses original pixels.
    const auto& options = impl_->options;
    const double scale = options.resize == 0 ? 1.0 : std::min(1.0, static_cast<double>(options.resize) / std::max(image_bgr.cols, image_bgr.rows));
    if (options.method == "sift") {
        cv::Mat work, gray;
        if (scale == 1.0) {
            work = image_bgr;
        } else {
            // Passing fx/fy preserves Python cv2.resize's interpolation scale
            // even when the shorter output dimension needs integer rounding.
            cv::resize(image_bgr, work, cv::Size(), scale, scale);
        }
        cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> points;
        impl_->sift->detectAndCompute(gray, cv::noArray(), points, result.descriptors);
        // OpenCV can retain tied responses beyond nfeatures; enforce the same
        // stable strongest-response budget as the Python reference.
        if (points.size() > static_cast<std::size_t>(options.max_keypoints)) {
            std::vector<int> retained(points.size());
            std::iota(retained.begin(), retained.end(), 0);
            std::stable_sort(retained.begin(), retained.end(), [&](int a, int b) {
                return points[static_cast<std::size_t>(a)].response > points[static_cast<std::size_t>(b)].response;
            });
            retained.resize(static_cast<std::size_t>(options.max_keypoints));
            std::vector<cv::KeyPoint> selected;
            cv::Mat descriptors(options.max_keypoints, 128, CV_32F);
            for (int i = 0; i < options.max_keypoints; ++i) {
                selected.push_back(points[static_cast<std::size_t>(retained[static_cast<std::size_t>(i)])]);
                result.descriptors.row(retained[static_cast<std::size_t>(i)]).copyTo(descriptors.row(i));
            }
            points = std::move(selected);
            result.descriptors = std::move(descriptors);
        }
        for (const auto& point : points) {
            // Per-axis actual scales reverse the rounded working-image dimensions.
            result.keypoints.emplace_back(point.pt.x * image_bgr.cols / work.cols, point.pt.y * image_bgr.rows / work.rows);
        }
        result.method = "sift";
        result.extraction_ms = elapsed_ms(started);
        return result;
    }
    // Match Kornia's long-side calculation, including truncation and the
    // order of floating-point operations used to compute the shorter side.
    const int target = options.resize == 0 ? std::max(image_bgr.cols, image_bgr.rows)
        : std::min(options.resize, std::max(image_bgr.cols, image_bgr.rows));
    const double aspect = static_cast<double>(image_bgr.cols) / image_bgr.rows;
    const cv::Size work_size = options.resize == 0 ? image_bgr.size()
        : (aspect < 1.0 ? cv::Size(static_cast<int>(target * aspect), target)
                        : cv::Size(target, static_cast<int>(target / aspect)));
    if (work_size.width < 5 || work_size.height < 5) {
        throw std::invalid_argument("ALIKED needs both resized image dimensions to be at least 5 pixels");
    }
    cv::Mat work = resize_rgb_float(image_bgr, work_size);
    const int pad_width = (32 - work.cols % 32) % 32;
    const int pad_height = (32 - work.rows % 32) % 32;
    const int left = pad_width / 2;  // Odd padding adds the extra pixel on the right.
    const int top = pad_height / 2;
    cv::Mat padded;
    cv::copyMakeBorder(work, padded, top, pad_height - top, left, pad_width - left, cv::BORDER_REPLICATE);
    const int spatial = padded.rows * padded.cols;
    std::vector<float> image(static_cast<std::size_t>(3) * spatial);
    for (int y = 0; y < padded.rows; ++y) {
        for (int x = 0; x < padded.cols; ++x) {
            const auto value = padded.at<cv::Vec3f>(y, x);
            for (int channel = 0; channel < 3; ++channel) {
                // ONNX uses NCHW; OpenCV stores an interleaved HWC image.
                image[static_cast<std::size_t>(channel) * spatial + y * padded.cols + x] = value[channel];
            }
        }
    }
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Tensor views borrow the vectors; these vectors outlive each synchronous Run.
    const std::array<std::int64_t, 4> image_shape{1, 3, padded.rows, padded.cols};
    auto input = Ort::Value::CreateTensor<float>(memory, image.data(), image.size(), image_shape.data(), image_shape.size());
    const char* image_names[] = {"image"};
    const char* dense_names[] = {"features", "scores"};
    auto dense = impl_->dense->Run(Ort::RunOptions{nullptr}, image_names, &input, 1, dense_names, 2);
    const auto* raw_scores = dense[1].GetTensorData<float>();
    const auto* raw_features = dense[0].GetTensorData<float>();
    cv::Mat scores(work_size, CV_32F);
    const int unpadded_spatial = work.rows * work.cols;
    std::vector<float> features(static_cast<std::size_t>(128) * unpadded_spatial);
    for (int y = 0; y < work.rows; ++y) {
        // Remove padding before DKD, which suppresses the true image border.
        std::copy_n(raw_scores + (y + top) * padded.cols + left, work.cols, scores.ptr<float>(y));
        for (int channel = 0; channel < 128; ++channel) {
            // Descriptor sampling needs the unpadded feature planes as well.
            std::copy_n(raw_features + static_cast<std::size_t>(channel) * spatial + (y + top) * padded.cols + left,
                        work.cols, features.data() + static_cast<std::size_t>(channel) * unpadded_spatial + y * work.cols);
        }
    }
    auto points = detect_aliked(scores, options.max_keypoints);
    result.method = "aliked-n16";
    if (!points.empty()) {
        std::vector<float> normalized(points.size() * 2);
        for (std::size_t i = 0; i < points.size(); ++i) {
            // ALIKED's descriptor head accepts align_corners=True normalized points.
            normalized[2*i] = 2.0F * points[i].x / (work.cols - 1) - 1.0F;
            normalized[2*i+1] = 2.0F * points[i].y / (work.rows - 1) - 1.0F;
            result.keypoints.emplace_back((points[i].x + 0.5F) * image_bgr.cols / work.cols - 0.5F,
                                          (points[i].y + 0.5F) * image_bgr.rows / work.rows - 0.5F);
        }
        const std::array<std::int64_t, 4> feature_shape{1, 128, work.rows, work.cols};
        // The sparse graph has a dynamic point dimension and returns N-by-128 rows.
        const std::array<std::int64_t, 2> point_shape{static_cast<std::int64_t>(points.size()), 2};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, features.data(), features.size(), feature_shape.data(), feature_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, normalized.data(), normalized.size(), point_shape.data(), point_shape.size()));
        const char* names[] = {"features", "keypoints"};
        const char* outputs[] = {"descriptors"};
        auto descriptor = impl_->descriptor->Run(Ort::RunOptions{nullptr}, names, inputs.data(), inputs.size(), outputs, 1);
        // Clone before releasing the ORT output so cached features own their data.
        result.descriptors = cv::Mat(static_cast<int>(points.size()), 128, CV_32F, descriptor[0].GetTensorMutableData<float>()).clone();
    }
    result.extraction_ms = elapsed_ms(started);
    return result;
}

MatchResult FeatureMatcher::match(const FeatureSet& reference, const FeatureSet& moving) {
    const auto started = Clock::now();
    MatchResult result;
    result.method = impl_->options.method;
    result.reference_keypoints = static_cast<int>(reference.keypoints.size());
    result.moving_keypoints = static_cast<int>(moving.keypoints.size());
    result.extraction_ms = reference.extraction_ms + moving.extraction_ms;
    // Both ALIKED routes can reuse the same cached extraction; SIFT cannot.
    const std::string expected = result.method == "sift" ? "sift" : "aliked-n16";
    if (reference.method != expected || moving.method != expected) {
        throw std::invalid_argument("Descriptor family does not match this feature matcher");
    }
    if (reference.keypoints.empty() || moving.keypoints.size() < 2) {
        // Two candidate descriptors are necessary for the ratio-test baseline.
        result.matching_ms = elapsed_ms(started);
        return result;
    }
    if (result.method != "lightglue") {
        std::vector<std::vector<cv::DMatch>> neighbors;
        // SIFT and ALIKED are float descriptors, so distance means Euclidean L2.
        cv::BFMatcher(cv::NORM_L2).knnMatch(reference.descriptors, moving.descriptors, neighbors, 2);
        for (const auto& pair : neighbors) {
            // Ambiguous nearest neighbors have distances too close to their runner-up.
            if (pair.size() == 2 && pair[0].distance < impl_->options.ratio * pair[1].distance) {
                result.reference_points.push_back(reference.keypoints[static_cast<std::size_t>(pair[0].queryIdx)]);
                result.moving_points.push_back(moving.keypoints[static_cast<std::size_t>(pair[0].trainIdx)]);
                result.scores.push_back(1.0F - pair[0].distance / std::max(pair[1].distance, 1e-12F));
            }
        }
        result.matching_ms = elapsed_ms(started);
        return result;
    }
    std::vector<int> indices0(reference.keypoints.size()), indices1(moving.keypoints.size());
    // Preserve original feature row numbers as later LightGlue layers prune points.
    std::iota(indices0.begin(), indices0.end(), 0);
    std::iota(indices1.begin(), indices1.end(), 0);
    auto normalize = [](const FeatureSet& features) {
        std::vector<float> points(features.keypoints.size()*2);
        // LightGlue centers pixels and divides both axes by half the longest side.
        const float scale = 0.5F * std::max(features.image_size.width, features.image_size.height);
        for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
            points[2*i] = (features.keypoints[i].x - 0.5F*features.image_size.width)/scale;
            points[2*i+1] = (features.keypoints[i].y - 0.5F*features.image_size.height)/scale;
        }
        return points;
    };
    auto keypoints0 = normalize(reference);
    auto keypoints1 = normalize(moving);
    std::vector<float> descriptors0(reference.descriptors.ptr<float>(), reference.descriptors.ptr<float>() + reference.descriptors.total());
    std::vector<float> descriptors1(moving.descriptors.ptr<float>(), moving.descriptors.ptr<float>() + moving.descriptors.total());
    const std::size_t original_total = indices0.size() + indices1.size();  // Early-stop denominator stays fixed.
    std::vector<float> final_scores;
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (int layer = 0; layer < 9; ++layer) {
        if (indices0.empty() || indices1.empty()) {
            result.matching_ms = elapsed_ms(started);
            return result;
        }
        // Layer zero projects ALIKED's 128 channels into LightGlue's 256 channels.
        const std::int64_t dimension = layer == 0 ? 128 : 256;
        const std::array<std::int64_t, 3> d0_shape{1, static_cast<std::int64_t>(indices0.size()), dimension};
        const std::array<std::int64_t, 3> d1_shape{1, static_cast<std::int64_t>(indices1.size()), dimension};
        const std::array<std::int64_t, 3> k0_shape{1, static_cast<std::int64_t>(indices0.size()), 2};
        const std::array<std::int64_t, 3> k1_shape{1, static_cast<std::int64_t>(indices1.size()), 2};
        std::vector<Ort::Value> inputs;
        // Two point sets can have different lengths; each keeps its own shape.
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, descriptors0.data(), descriptors0.size(), d0_shape.data(), d0_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, descriptors1.data(), descriptors1.size(), d1_shape.data(), d1_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, keypoints0.data(), keypoints0.size(), k0_shape.data(), k0_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, keypoints1.data(), keypoints1.size(), k1_shape.data(), k1_shape.size()));
        const char* names[] = {"descriptors0", "descriptors1", "keypoints0", "keypoints1"};
        const char* outputs[] = {"refined0", "refined1", "tokens0", "tokens1", "matchability0", "matchability1", "log_scores"};
        auto values = impl_->glue[static_cast<std::size_t>(layer)]->Run(Ort::RunOptions{nullptr}, names, inputs.data(), inputs.size(), outputs, 7);
        // Outputs include refinement, stopping confidence, pruning probability,
        // and the nondustbin log assignment matrix used if this is the last layer.
        descriptors0 = copy_tensor(values[0]);
        descriptors1 = copy_tensor(values[1]);
        final_scores = copy_tensor(values[6]);
        if (layer == 8) {
            // The final trained layer has no confidence head or subsequent pruning.
            break;
        }
        const auto token0 = copy_tensor(values[2]);
        const auto token1 = copy_tensor(values[3]);
        const float threshold = static_cast<float>(0.8 + 0.1*std::exp(-4.0*layer/9.0));
        std::size_t uncertain = 0;
        for (float token : token0) { uncertain += token < threshold; }
        for (float token : token1) { uncertain += token < threshold; }
        const float confident_ratio = 1.0F - static_cast<float>(uncertain) / static_cast<float>(original_total);
        // Previously pruned tokens count as confident, exactly as in upstream.
        if (confident_ratio > 0.95F) {
            break;
        }
        auto prune = [&](std::vector<int>& indices, std::vector<float>& descriptors,
                         std::vector<float>& points, const std::vector<float>& tokens, const Ort::Value& probabilities) {
            const float* matchability = probabilities.GetTensorData<float>();
            std::vector<int> keep, retained_indices;
            for (std::size_t i = 0; i < indices.size(); ++i) {
                // Low-confidence assignments remain available for later layers.
                if (matchability[i] > 0.01F || tokens[i] <= threshold) {
                    keep.push_back(static_cast<int>(i));
                    retained_indices.push_back(indices[i]);
                }
            }
            descriptors = gather_rows(descriptors, 256, keep);
            points = gather_rows(points, 2, keep);
            indices.swap(retained_indices);
        };
        // Native CPU inference follows the reference CPU pruning threshold (-1).
        prune(indices0, descriptors0, keypoints0, token0, values[4]);
        prune(indices1, descriptors1, keypoints1, token1, values[5]);
    }
    const std::size_t n = indices0.size(), m = indices1.size();
    if (!n || !m) {
        result.matching_ms = elapsed_ms(started);
        return result;
    }
    std::vector<std::size_t> row_best(n, 0), column_best(m, 0);
    // Compute row and column argmax indices. Strict comparison keeps the first
    // index on a tie, matching torch.max's deterministic argmax convention.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            if (final_scores[i*m+j] > final_scores[i*m+row_best[i]]) { row_best[i] = j; }
            if (final_scores[i*m+j] > final_scores[column_best[j]*m+j]) { column_best[j] = i; }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto j = row_best[i];
        const float score = std::exp(final_scores[i*m+j]);
        // A valid pair must be mutual and exceed the pretrained score threshold.
        if (column_best[j] == i && score > 0.1F) {
            result.reference_points.push_back(reference.keypoints[static_cast<std::size_t>(indices0[i])]);
            result.moving_points.push_back(moving.keypoints[static_cast<std::size_t>(indices1[j])]);
            result.scores.push_back(score);
        }
    }
    result.matching_ms = elapsed_ms(started);
    return result;
}

MatchResult FeatureMatcher::match_images(const cv::Mat& reference_bgr, const cv::Mat& moving_bgr) {
    // Call extract separately when a catalogue/reference will be queried repeatedly.
    return match(extract(reference_bgr), extract(moving_bgr));
}

}  // namespace robust_alignment
