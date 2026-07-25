// Dense Farneback, Sparse-to-Dense, and RLOF optical-flow examples.

#include "optical_flow.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace optical_flow {

namespace {

namespace fs = std::filesystem;

cv::Mat calculate_flow(
    Algorithm algorithm,
    const cv::Mat& previous_bgr,
    const cv::Mat& current_bgr) {
  cv::Mat flow;
  if (algorithm == Algorithm::kFarneback) {
    cv::Mat previous_gray;
    cv::Mat current_gray;
    cv::cvtColor(previous_bgr, previous_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(current_bgr, current_gray, cv::COLOR_BGR2GRAY);
    cv::calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        flow,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0);
  } else if (algorithm == Algorithm::kLucasKanadeDense) {
    cv::Mat previous_gray;
    cv::Mat current_gray;
    cv::cvtColor(previous_bgr, previous_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(current_bgr, current_gray, cv::COLOR_BGR2GRAY);
    cv::optflow::calcOpticalFlowSparseToDense(
        previous_gray,
        current_gray,
        flow);
  } else if (algorithm == Algorithm::kRlof) {
    cv::optflow::calcOpticalFlowDenseRLOF(
        previous_bgr,
        current_bgr,
        flow);
  } else {
    throw std::invalid_argument("Dense runner received a sparse algorithm");
  }
  return flow;
}

std::pair<cv::Mat, double> flow_to_bgr(const cv::Mat& flow) {
  std::vector<cv::Mat> channels;
  cv::split(flow, channels);

  cv::Mat magnitude;
  cv::Mat angle;
  cv::cartToPolar(channels[0], channels[1], magnitude, angle);
  const double mean_magnitude = cv::mean(magnitude)[0];

  cv::Mat hue_float = angle * (90.0 / CV_PI);
  cv::Mat hue;
  hue_float.convertTo(hue, CV_8U);

  cv::Mat value_float;
  cv::normalize(magnitude, value_float, 0.0, 255.0, cv::NORM_MINMAX);
  cv::Mat value;
  value_float.convertTo(value, CV_8U);

  cv::Mat saturation(
      magnitude.size(),
      CV_8U,
      cv::Scalar::all(255));
  std::vector<cv::Mat> hsv_channels{hue, saturation, value};
  cv::Mat hsv;
  cv::merge(hsv_channels, hsv);

  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  return {bgr, mean_magnitude};
}

void validate_summary(const Summary& summary) {
  if (summary.frame_pairs <= 0) {
    throw std::runtime_error(
        "No frame pairs were available for optical flow");
  }
  if (!std::isfinite(summary.mean_magnitude) ||
      summary.mean_magnitude <= 0.0) {
    throw std::runtime_error(
        "Expected finite nonzero motion, got " +
        std::to_string(summary.mean_magnitude));
  }
  const cv::Mat reloaded =
      cv::imread(summary.output_path.string(), cv::IMREAD_COLOR);
  if (reloaded.empty()) {
    throw std::runtime_error("Saved optical-flow visualization is unreadable");
  }
}

}  // namespace

std::string algorithm_name(Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kFarneback:
      return "farneback";
    case Algorithm::kLucasKanade:
      return "lucaskanade";
    case Algorithm::kLucasKanadeDense:
      return "lucaskanade_dense";
    case Algorithm::kRlof:
      return "rlof";
  }
  throw std::invalid_argument("Unknown optical-flow algorithm");
}

Summary run_dense_optical_flow(
    const RunOptions& options,
    Algorithm algorithm) {
  cv::VideoCapture capture(options.video_path.string());
  if (!capture.isOpened()) {
    throw std::runtime_error(
        "Unable to open input video: " + options.video_path.string());
  }

  cv::Mat previous_bgr;
  if (!capture.read(previous_bgr) || previous_bgr.empty()) {
    throw std::runtime_error(
        "Input video has no readable frames: " + options.video_path.string());
  }

  cv::Mat visualization;
  std::vector<double> magnitudes;
  int frame_pairs = 0;
  while (options.max_frames == 0 || frame_pairs < options.max_frames) {
    cv::Mat current_bgr;
    if (!capture.read(current_bgr) || current_bgr.empty()) {
      break;
    }

    const cv::Mat flow =
        calculate_flow(algorithm, previous_bgr, current_bgr);
    auto [flow_visualization, mean_magnitude] = flow_to_bgr(flow);
    visualization = std::move(flow_visualization);
    magnitudes.push_back(mean_magnitude);
    ++frame_pairs;

    if (options.show_windows) {
      cv::imshow("Input frame", current_bgr);
      cv::imshow("Dense optical flow", visualization);
      const int key = cv::waitKey(25);
      if (key == 27 || key == 'q') {
        break;
      }
    }
    previous_bgr = current_bgr;
  }

  capture.release();
  if (options.show_windows) {
    cv::destroyAllWindows();
  }
  if (frame_pairs == 0 || visualization.empty()) {
    throw std::runtime_error(
        "No frame pairs were available for optical flow");
  }

  fs::create_directories(options.output_dir);
  const fs::path output_path =
      options.output_dir /
      (algorithm_name(algorithm) + "-optical-flow.png");
  if (!cv::imwrite(output_path.string(), visualization)) {
    throw std::runtime_error(
        "Unable to write optical-flow image: " + output_path.string());
  }

  const double mean_magnitude =
      std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) /
      static_cast<double>(magnitudes.size());
  const Summary summary{frame_pairs, mean_magnitude, output_path};
  if (options.validate) {
    validate_summary(summary);
  }
  return summary;
}

}  // namespace optical_flow
