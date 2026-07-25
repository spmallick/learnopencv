// Sparse pyramidal Lucas-Kanade tracking for the optical-flow tutorial.

#include "optical_flow.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#endif
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace optical_flow {

namespace {

namespace fs = std::filesystem;

void validate_summary(const Summary& summary) {
  if (summary.frame_pairs <= 0) {
    throw std::runtime_error("No feature tracks were produced from the video");
  }
  if (!std::isfinite(summary.mean_magnitude) ||
      summary.mean_magnitude <= 0.0) {
    throw std::runtime_error(
        "Expected finite nonzero tracked motion, got " +
        std::to_string(summary.mean_magnitude));
  }
  const cv::Mat reloaded =
      cv::imread(summary.output_path.string(), cv::IMREAD_COLOR);
  if (reloaded.empty()) {
    throw std::runtime_error("Saved sparse-flow visualization is unreadable");
  }
}

}  // namespace

Summary run_lucas_kanade(const RunOptions& options) {
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

  cv::Mat previous_gray;
  cv::cvtColor(previous_bgr, previous_gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Point2f> previous_points;
  cv::goodFeaturesToTrack(
      previous_gray,
      previous_points,
      100,
      0.3,
      7.0,
      cv::noArray(),
      7);
  if (previous_points.empty()) {
    throw std::runtime_error(
        "No Shi-Tomasi features were found in the first frame");
  }

  cv::RNG generator(7);
  std::vector<cv::Scalar> colors;
  colors.reserve(100);
  for (int index = 0; index < 100; ++index) {
    colors.emplace_back(
        generator.uniform(0, 256),
        generator.uniform(0, 256),
        generator.uniform(0, 256));
  }

  cv::Mat trails = cv::Mat::zeros(previous_bgr.size(), previous_bgr.type());
  cv::Mat visualization = previous_bgr.clone();
  std::vector<double> magnitudes;
  int frame_pairs = 0;

  while (options.max_frames == 0 || frame_pairs < options.max_frames) {
    cv::Mat current_bgr;
    if (!capture.read(current_bgr) || current_bgr.empty()) {
      break;
    }

    cv::Mat current_gray;
    cv::cvtColor(current_bgr, current_gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> current_points;
    std::vector<unsigned char> status;
    std::vector<float> errors;
    cv::calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        current_points,
        status,
        errors,
        cv::Size(15, 15),
        2,
        cv::TermCriteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            10,
            0.03));

    std::vector<cv::Point2f> good_new;
    std::vector<double> frame_magnitudes;
    cv::Mat frame = current_bgr.clone();
    for (std::size_t index = 0; index < status.size(); ++index) {
      if (status[index] == 0U) {
        continue;
      }
      good_new.push_back(current_points[index]);
      const cv::Point2f displacement =
          current_points[index] - previous_points[index];
      frame_magnitudes.push_back(cv::norm(displacement));
      const cv::Scalar& color = colors[index % colors.size()];
      cv::line(
          trails,
          current_points[index],
          previous_points[index],
          color,
          2,
          cv::LINE_AA);
      cv::circle(
          frame,
          current_points[index],
          4,
          color,
          cv::FILLED,
          cv::LINE_AA);
    }
    if (good_new.empty()) {
      break;
    }

    const double frame_mean =
        std::accumulate(
            frame_magnitudes.begin(),
            frame_magnitudes.end(),
            0.0) /
        static_cast<double>(frame_magnitudes.size());
    magnitudes.push_back(frame_mean);
    cv::add(frame, trails, visualization);
    ++frame_pairs;

    if (options.show_windows) {
      cv::imshow("Sparse Lucas-Kanade optical flow", visualization);
      const int key = cv::waitKey(25);
      if (key == 27 || key == 'q') {
        break;
      }
      if (key == 'c') {
        trails.setTo(cv::Scalar::all(0));
      }
    }

    previous_gray = current_gray;
    previous_points = std::move(good_new);
  }

  capture.release();
  if (options.show_windows) {
    cv::destroyAllWindows();
  }
  if (frame_pairs == 0 || magnitudes.empty()) {
    throw std::runtime_error("No feature tracks were produced from the video");
  }

  fs::create_directories(options.output_dir);
  const fs::path output_path =
      options.output_dir / "lucaskanade-optical-flow.png";
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
