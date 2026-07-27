#ifndef LEARNOPENCV_FACE_DETECTION_HPP
#define LEARNOPENCV_FACE_DETECTION_HPP

// Shared detector contracts keep the primary and comparison examples aligned.

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>

#if CV_VERSION_MAJOR < 5
#include <opencv2/objdetect.hpp>
#endif

#ifdef FACE_WITH_DLIB
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace learnopencv::face {

struct Detection {
    cv::Rect2f box;
    std::array<cv::Point2f, 5> landmarks{};
    float score{};
    bool has_landmarks{};
};

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::string name() const = 0;
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
};

inline std::pair<int, int> backendTarget(const std::string& device) {
    if (device == "cpu") {
        return {
            cv::dnn::DNN_BACKEND_DEFAULT,
            cv::dnn::DNN_TARGET_CPU,
        };
    }
    if (device == "cuda") {
        return {
            cv::dnn::DNN_BACKEND_CUDA,
            cv::dnn::DNN_TARGET_CUDA,
        };
    }
    throw std::invalid_argument(
        "Unsupported device '" + device + "'. Choose 'cpu' or 'cuda'.");
}

class YuNetDetector final : public Detector {
public:
    explicit YuNetDetector(
        const std::filesystem::path& model_path,
        float score_threshold = 0.7F,
        float nms_threshold = 0.3F,
        int top_k = 5000,
        const std::string& device = "cpu") {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error(
                "YuNet model not found: " + model_path.string() +
                ". Run download_models.py first.");
        }
        if (score_threshold < 0.0F || score_threshold > 1.0F) {
            throw std::invalid_argument(
                "YuNet score threshold must be between 0 and 1.");
        }
        if (nms_threshold < 0.0F || nms_threshold > 1.0F) {
            throw std::invalid_argument(
                "YuNet NMS threshold must be between 0 and 1.");
        }
        if (top_k <= 0) {
            throw std::invalid_argument("YuNet top-k must be positive.");
        }

        const auto [backend_id, target_id] = backendTarget(device);
        // The 2026may model has dynamic height and width. A valid creation size
        // is still required, then detect() updates it for every real frame.
        model_ = cv::FaceDetectorYN::create(
            model_path.string(),
            "",
            cv::Size(320, 320),
            score_threshold,
            nms_threshold,
            top_k,
            backend_id,
            target_id);
        if (model_.empty()) {
            throw std::runtime_error(
                "OpenCV could not create FaceDetectorYN from: " +
                model_path.string());
        }
    }

    std::string name() const override {
        return "YuNet";
    }

    std::vector<Detection> detect(const cv::Mat& frame) override {
        if (frame.empty()) {
            throw std::invalid_argument("Cannot run YuNet on an empty frame.");
        }
        model_->setInputSize(frame.size());
        cv::Mat faces;
        model_->detect(frame, faces);
        if (faces.empty()) {
            return {};
        }
        if (faces.type() != CV_32F || faces.cols != 15) {
            throw std::runtime_error(
                "FaceDetectorYN returned an unexpected output shape or type.");
        }

        std::vector<Detection> detections;
        detections.reserve(static_cast<std::size_t>(faces.rows));
        for (int row_index = 0; row_index < faces.rows; ++row_index) {
            const float* row = faces.ptr<float>(row_index);
            Detection detection;
            detection.box = cv::Rect2f(row[0], row[1], row[2], row[3]);
            for (std::size_t index = 0; index < detection.landmarks.size(); ++index) {
                detection.landmarks[index] = cv::Point2f(
                    row[4 + 2 * index],
                    row[5 + 2 * index]);
            }
            detection.score = row[14];
            detection.has_landmarks = true;
            detections.push_back(detection);
        }
        return detections;
    }

private:
    cv::Ptr<cv::FaceDetectorYN> model_;
};

#if CV_VERSION_MAJOR < 5
class HaarDetector final : public Detector {
public:
    explicit HaarDetector(
        const std::filesystem::path& cascade_path,
        int resize_height = 300)
        : resize_height_(resize_height) {
        if (!std::filesystem::is_regular_file(cascade_path)) {
            throw std::runtime_error(
                "Haar cascade not found: " + cascade_path.string());
        }
        if (resize_height_ <= 0) {
            throw std::invalid_argument(
                "Haar resize height must be positive.");
        }
        if (!cascade_.load(cascade_path.string()) || cascade_.empty()) {
            throw std::runtime_error(
                "OpenCV could not load Haar cascade: " +
                cascade_path.string());
        }
    }

    std::string name() const override {
        return "Haar";
    }

    std::vector<Detection> detect(const cv::Mat& frame) override {
        if (frame.empty()) {
            throw std::invalid_argument("Cannot run Haar on an empty frame.");
        }
        const int resized_width = std::max(
            1,
            cvRound(
                static_cast<double>(frame.cols) *
                static_cast<double>(resize_height_) /
                static_cast<double>(frame.rows)));
        cv::Mat small;
        cv::resize(
            frame,
            small,
            cv::Size(resized_width, resize_height_));
        cv::Mat gray;
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> boxes;
        cascade_.detectMultiScale(gray, boxes);

        const float scale_x =
            static_cast<float>(frame.cols) /
            static_cast<float>(resized_width);
        const float scale_y =
            static_cast<float>(frame.rows) /
            static_cast<float>(resize_height_);
        std::vector<Detection> detections;
        detections.reserve(boxes.size());
        for (const cv::Rect& box : boxes) {
            Detection detection;
            detection.box = cv::Rect2f(
                static_cast<float>(box.x) * scale_x,
                static_cast<float>(box.y) * scale_y,
                static_cast<float>(box.width) * scale_x,
                static_cast<float>(box.height) * scale_y);
            detection.score = 1.0F;
            detection.has_landmarks = false;
            detections.push_back(detection);
        }
        return detections;
    }

private:
    cv::CascadeClassifier cascade_;
    int resize_height_;
};
#endif

#ifdef FACE_WITH_DLIB
class DlibHogDetector final : public Detector {
public:
    explicit DlibHogDetector(int resize_height = 300)
        : detector_(dlib::get_frontal_face_detector()),
          resize_height_(resize_height) {
        if (resize_height_ <= 0) {
            throw std::invalid_argument(
                "dlib HOG resize height must be positive.");
        }
    }

    std::string name() const override {
        return "dlib HOG";
    }

    std::vector<Detection> detect(const cv::Mat& frame) override {
        if (frame.empty()) {
            throw std::invalid_argument(
                "Cannot run dlib HOG on an empty frame.");
        }
        const int resized_width = std::max(
            1,
            cvRound(
                static_cast<double>(frame.cols) *
                static_cast<double>(resize_height_) /
                static_cast<double>(frame.rows)));
        cv::Mat small;
        cv::resize(
            frame,
            small,
            cv::Size(resized_width, resize_height_));
        dlib::cv_image<dlib::bgr_pixel> dlib_image(small);
        const std::vector<dlib::rectangle> boxes = detector_(dlib_image);

        const float scale_x =
            static_cast<float>(frame.cols) /
            static_cast<float>(resized_width);
        const float scale_y =
            static_cast<float>(frame.rows) /
            static_cast<float>(resize_height_);
        std::vector<Detection> detections;
        detections.reserve(boxes.size());
        for (const dlib::rectangle& box : boxes) {
            const float x1 = static_cast<float>(box.left()) * scale_x;
            const float y1 = static_cast<float>(box.top()) * scale_y;
            const float x2 =
                static_cast<float>(box.right() + 1) * scale_x;
            const float y2 =
                static_cast<float>(box.bottom() + 1) * scale_y;
            Detection detection;
            detection.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            detection.score = 1.0F;
            detection.has_landmarks = false;
            detections.push_back(detection);
        }
        return detections;
    }

private:
    dlib::frontal_face_detector detector_;
    int resize_height_;
};
#endif

inline cv::Rect clippedBox(
    const Detection& detection,
    const cv::Size& frame_size) {
    int x1 = static_cast<int>(std::floor(detection.box.x));
    int y1 = static_cast<int>(std::floor(detection.box.y));
    int x2 = static_cast<int>(
        std::ceil(detection.box.x + detection.box.width));
    int y2 = static_cast<int>(
        std::ceil(detection.box.y + detection.box.height));
    x1 = std::clamp(x1, 0, std::max(0, frame_size.width - 1));
    y1 = std::clamp(y1, 0, std::max(0, frame_size.height - 1));
    x2 = std::clamp(x2, x1 + 1, frame_size.width);
    y2 = std::clamp(y2, y1 + 1, frame_size.height);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

inline void validate(
    const cv::Mat& frame,
    const std::vector<Detection>& detections) {
    if (frame.empty()) {
        throw std::runtime_error("Validation received an empty frame.");
    }
    for (const Detection& detection : detections) {
        if (!std::isfinite(detection.box.x) ||
            !std::isfinite(detection.box.y) ||
            !std::isfinite(detection.box.width) ||
            !std::isfinite(detection.box.height) ||
            !std::isfinite(detection.score)) {
            throw std::runtime_error(
                "Face detection contains a non-finite value.");
        }
        if (detection.box.width <= 0.0F ||
            detection.box.height <= 0.0F) {
            throw std::runtime_error(
                "Face detection has a non-positive box size.");
        }
        if (detection.score < 0.0F || detection.score > 1.0F) {
            throw std::runtime_error(
                "Face detection score is outside [0, 1].");
        }
        const cv::Rect clipped = clippedBox(detection, frame.size());
        if (clipped.width <= 0 || clipped.height <= 0 ||
            clipped.x < 0 || clipped.y < 0 ||
            clipped.br().x > frame.cols ||
            clipped.br().y > frame.rows) {
            throw std::runtime_error(
                "Face detection cannot be clipped to valid image bounds.");
        }
        if (detection.has_landmarks) {
            for (const cv::Point2f& landmark : detection.landmarks) {
                if (!std::isfinite(landmark.x) ||
                    !std::isfinite(landmark.y)) {
                    throw std::runtime_error(
                        "Face landmark contains a non-finite value.");
                }
            }
        }
    }
}

inline cv::Mat draw(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& label) {
    if (frame.empty()) {
        throw std::invalid_argument("Cannot draw on an empty frame.");
    }
    cv::Mat output = frame.clone();
    static const std::array<cv::Scalar, 5> colors{{
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
    }};

    for (const Detection& detection : detections) {
        const cv::Rect box = clippedBox(detection, frame.size());
        cv::rectangle(output, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(
            output,
            cv::format("%.3f", detection.score),
            cv::Point(box.x, std::max(14, box.y + 14)),
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1,
            cv::LINE_AA);

        if (detection.has_landmarks) {
            for (std::size_t index = 0; index < colors.size(); ++index) {
                const cv::Point point{
                    cvRound(detection.landmarks[index].x),
                    cvRound(detection.landmarks[index].y),
                };
                if (point.x >= 0 && point.x < frame.cols &&
                    point.y >= 0 && point.y < frame.rows) {
                    cv::circle(
                        output,
                        point,
                        2,
                        colors[index],
                        2,
                        cv::LINE_AA);
                }
            }
        }
    }

    cv::rectangle(
        output,
        cv::Rect(0, 0, output.cols, std::min(34, output.rows)),
        cv::Scalar(0, 0, 0),
        cv::FILLED);
    cv::putText(
        output,
        label + ": " + std::to_string(detections.size()) + " face(s)",
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);
    return output;
}

inline cv::Mat joinPanels(const std::vector<cv::Mat>& panels) {
    if (panels.empty()) {
        throw std::invalid_argument("No comparison panels were provided.");
    }
    cv::Mat comparison;
    cv::hconcat(panels, comparison);
    return comparison;
}

}  // namespace learnopencv::face

#endif  // LEARNOPENCV_FACE_DETECTION_HPP
