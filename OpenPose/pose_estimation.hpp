#ifndef LEARNOPENCV_POSE_ESTIMATION_HPP
#define LEARNOPENCV_POSE_ESTIMATION_HPP

// This header keeps image and video examples on exactly the same preprocessing,
// inference, decoding, drawing, and validation path.

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace learnopencv::pose {

constexpr int kLandmarkCount = 33;
constexpr int kModelLandmarkCount = 39;
constexpr int kModelInputSize = 256;

// MediaPipe Pose connections use the official 33-landmark numbering.
inline constexpr std::array<std::array<int, 2>, 35> kPoseEdges{{
    {{0, 1}},   {{1, 2}},   {{2, 3}},   {{3, 7}},   {{0, 4}},
    {{4, 5}},   {{5, 6}},   {{6, 8}},   {{9, 10}},  {{11, 12}},
    {{11, 13}}, {{13, 15}}, {{15, 17}}, {{15, 19}}, {{15, 21}},
    {{17, 19}}, {{12, 14}}, {{14, 16}}, {{16, 18}}, {{16, 20}},
    {{16, 22}}, {{18, 20}}, {{11, 23}}, {{12, 24}}, {{23, 24}},
    {{23, 25}}, {{25, 27}}, {{27, 29}}, {{27, 31}}, {{29, 31}},
    {{24, 26}}, {{26, 28}}, {{28, 30}}, {{28, 32}}, {{30, 32}},
}};

struct Landmark {
    float x{};
    float y{};
    float z{};
    float visibility{};
    float presence{};
};

struct PoseResult {
    std::array<Landmark, kLandmarkCount> landmarks{};
    float confidence{};
};

struct DrawMetrics {
    int visible_count{};
    int edge_count{};
};

struct SquareTransform {
    int side{};
    int left{};
    int top{};
};

inline float sigmoid(float value) {
    // Clamping avoids overflow without changing probabilities at useful logits.
    const float clipped = std::clamp(value, -60.0F, 60.0F);
    return 1.0F / (1.0F + std::exp(-clipped));
}

inline void configureBackend(cv::dnn::Net& net, const std::string& device) {
    if (device == "cpu") {
        // DEFAULT selects the compatible graph engine in OpenCV 4.14 and 5.0.
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return;
    }
    if (device == "cuda") {
        // This path fails clearly when OpenCV was built without CUDA DNN support.
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        return;
    }
    throw std::invalid_argument(
        "Unsupported device '" + device + "'. Choose 'cpu' or 'cuda'.");
}

inline cv::Mat preprocess(
    const cv::Mat& frame,
    SquareTransform& transform) {
    if (frame.empty()) {
        throw std::invalid_argument("Cannot preprocess an empty frame.");
    }

    transform.side = std::max(frame.rows, frame.cols);
    transform.left = (transform.side - frame.cols) / 2;
    const int right = transform.side - frame.cols - transform.left;
    transform.top = (transform.side - frame.rows) / 2;
    const int bottom = transform.side - frame.rows - transform.top;

    // Letterboxing preserves the full person without stretching the image.
    cv::Mat square;
    cv::copyMakeBorder(
        frame,
        square,
        transform.top,
        bottom,
        transform.left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));

    cv::Mat resized;
    cv::resize(
        square,
        resized,
        cv::Size(kModelInputSize, kModelInputSize),
        0.0,
        0.0,
        cv::INTER_AREA);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    if (!normalized.isContinuous()) {
        normalized = normalized.clone();
    }

    // The pinned OpenCV Zoo model consumes NHWC input. The view remains valid
    // because normalized owns the same continuous bytes through this return.
    const int sizes[] = {1, kModelInputSize, kModelInputSize, 3};
    return cv::Mat(4, sizes, CV_32F, normalized.data).clone();
}

inline const cv::Mat& findOutput(
    const std::vector<cv::Mat>& outputs,
    std::size_t element_count) {
    const cv::Mat* match = nullptr;
    for (const cv::Mat& output : outputs) {
        if (output.total() == element_count) {
            if (match != nullptr) {
                throw std::runtime_error(
                    "Pose model returned multiple outputs with " +
                    std::to_string(element_count) + " values.");
            }
            match = &output;
        }
    }
    if (match == nullptr) {
        throw std::runtime_error(
            "Pose model did not return an output with " +
            std::to_string(element_count) + " values.");
    }
    return *match;
}

inline PoseResult decode(
    const std::vector<cv::Mat>& outputs,
    const SquareTransform& transform) {
    const cv::Mat& raw_landmarks =
        findOutput(outputs, static_cast<std::size_t>(kModelLandmarkCount * 5));
    const cv::Mat& raw_confidence = findOutput(outputs, 1U);
    if (raw_landmarks.type() != CV_32F || raw_confidence.type() != CV_32F) {
        throw std::runtime_error("Pose model returned a non-float output.");
    }

    const float* values = raw_landmarks.ptr<float>();
    const float scale =
        static_cast<float>(transform.side) / static_cast<float>(kModelInputSize);
    PoseResult result;
    for (int index = 0; index < kLandmarkCount; ++index) {
        const int offset = index * 5;
        result.landmarks[static_cast<std::size_t>(index)] = Landmark{
            values[offset] * scale - static_cast<float>(transform.left),
            values[offset + 1] * scale - static_cast<float>(transform.top),
            values[offset + 2] * scale,
            sigmoid(values[offset + 3]),
            sigmoid(values[offset + 4]),
        };
    }
    result.confidence = raw_confidence.ptr<float>()[0];
    return result;
}

class PoseEstimator {
public:
    explicit PoseEstimator(
        const std::filesystem::path& model_path,
        const std::string& device = "cpu") {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error(
                "Pose model not found: " + model_path.string() +
                ". Run download_models.py first.");
        }
        net_ = cv::dnn::readNet(model_path.string());
        if (net_.empty()) {
            throw std::runtime_error(
                "OpenCV could not load pose model: " + model_path.string());
        }
        configureBackend(net_, device);
    }

    PoseResult infer(const cv::Mat& frame) {
        SquareTransform transform;
        const cv::Mat blob = preprocess(frame, transform);
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        return decode(outputs, transform);
    }

private:
    cv::dnn::Net net_;
};

inline DrawMetrics draw(
    cv::Mat& output,
    const PoseResult& result,
    float score_threshold) {
    if (output.empty()) {
        throw std::invalid_argument("Cannot draw on an empty frame.");
    }

    std::array<cv::Point, kLandmarkCount> points{};
    std::array<bool, kLandmarkCount> keep{};
    DrawMetrics metrics;
    for (int index = 0; index < kLandmarkCount; ++index) {
        const Landmark& landmark =
            result.landmarks[static_cast<std::size_t>(index)];
        const cv::Point point{
            cvRound(landmark.x),
            cvRound(landmark.y),
        };
        const bool in_bounds =
            point.x >= 0 && point.x < output.cols &&
            point.y >= 0 && point.y < output.rows;
        const bool visible =
            landmark.visibility >= score_threshold &&
            landmark.presence >= score_threshold &&
            in_bounds;
        points[static_cast<std::size_t>(index)] = point;
        keep[static_cast<std::size_t>(index)] = visible;
        metrics.visible_count += visible ? 1 : 0;
    }

    for (const auto& edge : kPoseEdges) {
        const int start = edge[0];
        const int end = edge[1];
        if (keep[static_cast<std::size_t>(start)] &&
            keep[static_cast<std::size_t>(end)]) {
            cv::line(
                output,
                points[static_cast<std::size_t>(start)],
                points[static_cast<std::size_t>(end)],
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA);
            ++metrics.edge_count;
        }
    }

    for (int index = 0; index < kLandmarkCount; ++index) {
        if (keep[static_cast<std::size_t>(index)]) {
            cv::circle(
                output,
                points[static_cast<std::size_t>(index)],
                4,
                cv::Scalar(0, 0, 255),
                cv::FILLED,
                cv::LINE_AA);
        }
    }

    cv::putText(
        output,
        cv::format("Pose confidence: %.3f", result.confidence),
        cv::Point(12, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA);
    return metrics;
}

inline void validate(
    const cv::Mat& frame,
    const PoseResult& result,
    const DrawMetrics& metrics) {
    if (frame.empty()) {
        throw std::runtime_error("Validation received an empty frame.");
    }
    if (!std::isfinite(result.confidence) ||
        result.confidence < 0.0F ||
        result.confidence > 1.0F) {
        throw std::runtime_error("Pose confidence is outside [0, 1].");
    }
    for (const Landmark& landmark : result.landmarks) {
        if (!std::isfinite(landmark.x) ||
            !std::isfinite(landmark.y) ||
            !std::isfinite(landmark.z) ||
            !std::isfinite(landmark.visibility) ||
            !std::isfinite(landmark.presence)) {
            throw std::runtime_error("Pose output contains a non-finite value.");
        }
    }
    if (metrics.visible_count < 0 || metrics.visible_count > kLandmarkCount) {
        throw std::runtime_error("Visible landmark count is invalid.");
    }
    if (metrics.edge_count < 0 ||
        metrics.edge_count > static_cast<int>(kPoseEdges.size())) {
        throw std::runtime_error("Skeleton edge count is invalid.");
    }
}

}  // namespace learnopencv::pose

#endif  // LEARNOPENCV_POSE_ESTIMATION_HPP
