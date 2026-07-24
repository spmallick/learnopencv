/*
Copyright (c) 2014, Nghia Ho
Copyright (c) 2019, Big Vision LLC (Satya Mallick)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the conditions in the original LearnOpenCV example
are met. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
*/

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultSmoothingRadius = 50;

struct Options {
  std::filesystem::path input =
      std::filesystem::path(VIDEO_STABILIZATION_SOURCE_DIR) / "video.mp4";
  std::filesystem::path output_dir =
      std::filesystem::path(VIDEO_STABILIZATION_SOURCE_DIR) / "output";
  std::string output_name = "video_out.mp4";
  int smoothing_radius = kDefaultSmoothingRadius;
  bool display = true;
  bool validate = false;
};

struct Transform {
  double dx = 0.0;
  double dy = 0.0;
  double angle = 0.0;
};

struct Trajectory {
  double x = 0.0;
  double y = 0.0;
  double angle = 0.0;
};

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --input PATH             Input video (default: bundled video.mp4)\n"
      << "  --output-dir PATH        Output directory (default: output)\n"
      << "  --output-name NAME       Output filename (default: video_out.mp4)\n"
      << "  --smoothing-radius N     Moving-average radius (default: 50)\n"
      << "  --no-display             Disable the preview window\n"
      << "  --validate               Validate the generated video\n"
      << "  --help                   Show this help\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto require_value = [&](const std::string& name) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value.");
      }
      return argv[++index];
    };

    if (argument == "--input") {
      options.input = require_value(argument);
    } else if (argument == "--output-dir") {
      options.output_dir = require_value(argument);
    } else if (argument == "--output-name") {
      options.output_name = require_value(argument);
    } else if (argument == "--smoothing-radius") {
      const std::string value = require_value(argument);
      std::size_t consumed = 0;
      options.smoothing_radius = std::stoi(value, &consumed);
      if (consumed != value.size() || options.smoothing_radius < 0) {
        throw std::invalid_argument(
            "The smoothing radius must be a non-negative integer.");
      }
    } else if (argument == "--no-display") {
      options.display = false;
    } else if (argument == "--validate") {
      options.validate = true;
    } else if (argument == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown option: " + argument);
    }
  }
  return options;
}

cv::Mat identity_transform() {
  return (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

std::vector<Transform> estimate_transforms(
    cv::VideoCapture& capture, std::vector<std::size_t>& tracked_counts) {
  cv::Mat previous;
  if (!capture.read(previous) || previous.empty()) {
    throw std::runtime_error(
        "The input video does not contain a readable frame.");
  }

  cv::Mat previous_gray;
  cv::cvtColor(previous, previous_gray, cv::COLOR_BGR2GRAY);
  cv::Mat last_transform = identity_transform();
  std::vector<Transform> transforms;

  while (true) {
    cv::Mat current;
    if (!capture.read(current) || current.empty()) {
      break;
    }

    cv::Mat current_gray;
    cv::cvtColor(current, current_gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> previous_points;
    cv::goodFeaturesToTrack(
        previous_gray, previous_points, 200, 0.01, 30.0, cv::noArray(), 3);

    cv::Mat transform;
    std::size_t tracked_count = 0;
    if (previous_points.size() >= 3U) {
      std::vector<cv::Point2f> current_points;
      std::vector<unsigned char> status;
      std::vector<float> errors;
      cv::calcOpticalFlowPyrLK(
          previous_gray, current_gray, previous_points, current_points, status,
          errors);

      std::vector<cv::Point2f> valid_previous;
      std::vector<cv::Point2f> valid_current;
      valid_previous.reserve(status.size());
      valid_current.reserve(status.size());
      for (std::size_t index = 0; index < status.size(); ++index) {
        if (status[index] != 0U) {
          valid_previous.push_back(previous_points[index]);
          valid_current.push_back(current_points[index]);
        }
      }
      tracked_count = valid_previous.size();
      if (tracked_count >= 3U) {
        // This shared OpenCV 4/5 API models translation, rotation, and uniform
        // scale without introducing the shear of a full affine transform.
        transform =
            cv::estimateAffinePartial2D(valid_previous, valid_current);
      }
    }

    // Blurred or textureless frame pairs may not have three usable matches.
    // Reuse the most recent motion estimate to prevent a discontinuous jump.
    if (transform.empty() || !cv::checkRange(transform)) {
      transform = last_transform.clone();
    } else {
      transform.convertTo(transform, CV_64F);
      last_transform = transform.clone();
    }

    transforms.push_back(
        {transform.at<double>(0, 2), transform.at<double>(1, 2),
         std::atan2(transform.at<double>(1, 0),
                    transform.at<double>(0, 0))});
    tracked_counts.push_back(tracked_count);
    previous_gray = current_gray;
  }

  if (transforms.empty()) {
    throw std::runtime_error(
        "The input video must contain at least two frames.");
  }
  return transforms;
}

std::vector<Trajectory> cumulative_trajectory(
    const std::vector<Transform>& transforms) {
  std::vector<Trajectory> trajectory;
  trajectory.reserve(transforms.size());
  Trajectory accumulated;
  for (const Transform& transform : transforms) {
    accumulated.x += transform.dx;
    accumulated.y += transform.dy;
    accumulated.angle += transform.angle;
    trajectory.push_back(accumulated);
  }
  return trajectory;
}

std::vector<Trajectory> smooth_trajectory(
    const std::vector<Trajectory>& trajectory, int radius) {
  std::vector<Trajectory> smoothed;
  smoothed.reserve(trajectory.size());
  for (std::size_t index = 0; index < trajectory.size(); ++index) {
    Trajectory sum;
    for (int offset = -radius; offset <= radius; ++offset) {
      const auto unbounded =
          static_cast<long long>(index) + static_cast<long long>(offset);
      // Clamp out-of-range samples to the nearest endpoint. This matches the
      // edge-padded NumPy filter and keeps a fixed-width window at every frame.
      const auto candidate = std::clamp(
          unbounded, 0LL,
          static_cast<long long>(trajectory.size()) - 1LL);
      const Trajectory& value =
          trajectory[static_cast<std::size_t>(candidate)];
      sum.x += value.x;
      sum.y += value.y;
      sum.angle += value.angle;
    }
    const double count = static_cast<double>(2 * radius + 1);
    smoothed.push_back(
        {sum.x / count, sum.y / count, sum.angle / count});
  }
  return smoothed;
}

cv::Mat transform_matrix(const Transform& transform) {
  const double cosine = std::cos(transform.angle);
  const double sine = std::sin(transform.angle);
  return (cv::Mat_<double>(2, 3) << cosine, -sine, transform.dx, sine,
          cosine, transform.dy);
}

void fix_border(cv::Mat& frame) {
  const cv::Point2f center(
      static_cast<float>(frame.cols) / 2.0F,
      static_cast<float>(frame.rows) / 2.0F);
  const cv::Mat transform = cv::getRotationMatrix2D(center, 0.0, 1.04);
  cv::warpAffine(frame, frame, transform, frame.size());
}

int run(const Options& options) {
  cv::VideoCapture capture(options.input.string());
  if (!capture.isOpened()) {
    throw std::runtime_error(
        "Could not open input video: " + options.input.string());
  }

  const int width =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = capture.get(cv::CAP_PROP_FPS);
  if (width <= 0 || height <= 0 || !std::isfinite(fps) || fps <= 0.0) {
    throw std::runtime_error(
        "The input video has invalid dimensions or frame rate.");
  }

  std::vector<std::size_t> tracked_counts;
  const std::vector<Transform> transforms =
      estimate_transforms(capture, tracked_counts);
  const std::vector<Trajectory> trajectory =
      cumulative_trajectory(transforms);
  const std::vector<Trajectory> smoothed =
      smooth_trajectory(trajectory, options.smoothing_radius);

  std::vector<Transform> corrected;
  corrected.reserve(transforms.size());
  for (std::size_t index = 0; index < transforms.size(); ++index) {
    corrected.push_back(
        {transforms[index].dx + smoothed[index].x - trajectory[index].x,
         transforms[index].dy + smoothed[index].y - trajectory[index].y,
         transforms[index].angle + smoothed[index].angle -
             trajectory[index].angle});
  }

  std::filesystem::create_directories(options.output_dir);
  const std::filesystem::path output_path =
      options.output_dir / options.output_name;
  int output_width = 2 * width;
  int output_height = height;
  if (output_width > 1920) {
    output_width /= 2;
    output_height /= 2;
  }
  cv::VideoWriter writer(
      output_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
      cv::Size(output_width, output_height));
  if (!writer.isOpened()) {
    throw std::runtime_error(
        "Could not open output video: " + output_path.string());
  }

  capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
  std::size_t written_frames = 0;
  for (const Transform& transform : corrected) {
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
      break;
    }

    cv::Mat stabilized;
    cv::warpAffine(
        frame, stabilized, transform_matrix(transform), frame.size());
    fix_border(stabilized);

    cv::Mat comparison;
    cv::hconcat(frame, stabilized, comparison);
    if (comparison.cols != output_width ||
        comparison.rows != output_height) {
      cv::resize(
          comparison, comparison, cv::Size(output_width, output_height));
    }
    writer.write(comparison);
    ++written_frames;

    if (options.display) {
      cv::imshow("Before and After", comparison);
      if ((cv::waitKey(1) & 0xFF) == 27) {
        break;
      }
    }
  }
  capture.release();
  writer.release();
  cv::destroyAllWindows();

  if (options.validate) {
    if (written_frames != transforms.size()) {
      throw std::runtime_error(
          "The number of output frames did not match the transforms.");
    }
    if (!std::filesystem::is_regular_file(output_path) ||
        std::filesystem::file_size(output_path) == 0U) {
      throw std::runtime_error("The output video is missing or empty.");
    }
    cv::VideoCapture check(output_path.string());
    cv::Mat first_output_frame;
    if (!check.read(first_output_frame) || first_output_frame.empty()) {
      throw std::runtime_error(
          "OpenCV could not decode the generated video.");
    }
    if (first_output_frame.cols != output_width ||
        first_output_frame.rows != output_height) {
      throw std::runtime_error(
          "The generated video has unexpected dimensions.");
    }
    const double tracked_sum = std::accumulate(
        tracked_counts.begin(), tracked_counts.end(), 0.0);
    const double tracked_mean =
        tracked_sum / static_cast<double>(tracked_counts.size());
    std::cout << "VALIDATION PASSED: " << written_frames << " frames, "
              << output_width << 'x' << output_height
              << ", mean tracked points " << tracked_mean << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
