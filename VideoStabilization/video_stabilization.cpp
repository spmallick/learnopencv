/*
Copyright (c) 2014, Nghia Ho
Copyright (c) 2019, Big Vision LLC (Satya Mallick)
https://bigvision.ai  contact@bigvision.ai
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#if CV_VERSION_MAJOR >= 5
// OpenCV 5 moved goodFeaturesToTrack and affine estimation into new modules.
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#else
// OpenCV 4 declares estimateAffinePartial2D in the calib3d module.
#include <opencv2/calib3d.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// A radius of 50 creates a 101-frame smoothing window around each sample.
constexpr int kDefaultSmoothingRadius = 50;

// Options holds the complete command-line contract in one testable value.
struct Options {
  // Resolve bundled input and default output from the source directory supplied
  // by CMake, so execution never depends on the current working directory.
  std::filesystem::path input =
      std::filesystem::path(VIDEO_STABILIZATION_SOURCE_DIR) / "video.mp4";
  std::filesystem::path output_dir =
      std::filesystem::path(VIDEO_STABILIZATION_SOURCE_DIR) / "output";

  // Keep filename and tuning controls separate from their directory/path data.
  std::string output_name = "video_out.mp4";
  int smoothing_radius = kDefaultSmoothingRadius;

  // Interactive display is the tutorial default; CI can disable it and enable
  // semantic validation independently.
  bool display = true;
  bool validate = false;
};

// Transform represents pairwise camera motion in pixels and radians.
struct Transform {
  double dx = 0.0;
  double dy = 0.0;
  double angle = 0.0;
};

// Trajectory represents accumulated camera position in the same three axes.
struct Trajectory {
  double x = 0.0;
  double y = 0.0;
  double angle = 0.0;
};

void print_usage(const char* program) {
  // Keep the usage text aligned with the Python command-line interface.
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
  // Begin with documented defaults and replace only explicitly supplied values.
  Options options;

  // Parse each token once; value-taking options advance index inside the helper.
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];

    // Centralizing missing-value handling gives every option a clear error.
    const auto require_value = [&](const std::string& name) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value.");
      }
      return argv[++index];
    };

    // Convert path-like values into filesystem paths immediately.
    if (argument == "--input") {
      options.input = require_value(argument);
    } else if (argument == "--output-dir") {
      options.output_dir = require_value(argument);
    } else if (argument == "--output-name") {
      options.output_name = require_value(argument);
    } else if (argument == "--smoothing-radius") {
      // stoi reports numeric conversion errors; consumed rejects suffixes such
      // as "10frames" that would otherwise look partially valid.
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
      // Help is a successful terminal action, so it exits with status zero.
      print_usage(argv[0]);
      std::exit(0);
    } else {
      // Reject misspelled controls rather than silently running with defaults.
      throw std::invalid_argument("Unknown option: " + argument);
    }
  }

  // The caller receives a fully validated, immutable-by-convention snapshot.
  return options;
}

cv::Mat identity_transform() {
  // Mat::eye avoids the deprecated comma initializer and creates the required
  // two-row affine identity directly in double precision.
  return cv::Mat::eye(2, 3, CV_64F);
}

bool paths_refer_to_same_file(
    const std::filesystem::path& input,
    const std::filesystem::path& output) {
  // weakly_canonical resolves symlinks and normalizes a not-yet-created output.
  const std::filesystem::path normalized_input =
      std::filesystem::weakly_canonical(input);
  const std::filesystem::path normalized_output =
      std::filesystem::weakly_canonical(output);
  if (normalized_input == normalized_output) {
    return true;
  }

  // equivalent additionally detects an existing hard link to the input file.
  return std::filesystem::exists(input) &&
         std::filesystem::exists(output) &&
         std::filesystem::equivalent(input, output);
}

std::vector<Transform> estimate_transforms(
    cv::VideoCapture& capture, std::vector<std::size_t>& tracked_counts) {
  // The first decoded frame seeds the adjacent-frame comparison loop.
  cv::Mat previous;
  if (!capture.read(previous) || previous.empty()) {
    throw std::runtime_error(
        "The input video does not contain a readable frame.");
  }

  // Sparse feature detection and optical flow operate on grayscale intensities.
  cv::Mat previous_gray;
  cv::cvtColor(previous, previous_gray, cv::COLOR_BGR2GRAY);

  // Identity is a safe initial fallback for an untrackable first pair.
  cv::Mat last_transform = identity_transform();

  // Reserve dynamically because container metadata may overstate readable frames.
  std::vector<Transform> transforms;

  // Decode until the selected video backend reaches the readable stream end.
  while (true) {
    cv::Mat current;
    if (!capture.read(current) || current.empty()) {
      break;
    }

    // Only the newly decoded frame needs grayscale conversion each iteration.
    cv::Mat current_gray;
    cv::cvtColor(current, current_gray, cv::COLOR_BGR2GRAY);

    // Shi-Tomasi corners provide sparse, well-localized points for tracking.
    std::vector<cv::Point2f> previous_points;
    cv::goodFeaturesToTrack(
        previous_gray, previous_points, 200, 0.01, 30.0, cv::noArray(), 3);

    // An empty transform marks a pair for which no reliable model was estimated.
    cv::Mat transform;
    std::size_t tracked_count = 0;

    // A partial affine model requires at least three point correspondences.
    if (previous_points.size() >= 3U) {
      // Pyramidal Lucas-Kanade follows each corner into the current frame.
      std::vector<cv::Point2f> current_points;
      std::vector<unsigned char> status;
      std::vector<float> errors;
      cv::calcOpticalFlowPyrLK(
          previous_gray, current_gray, previous_points, current_points, status,
          errors);

      // Retain only tracks that OpenCV marks as successfully followed.
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

      // The count is both a model precondition and a useful quality diagnostic.
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
      // Normalize the matrix type before reading double-precision elements.
      transform.convertTo(transform, CV_64F);

      // Retain an independent matrix for a later frame pair's fallback.
      last_transform = transform.clone();
    }

    // Extract translation from the last column and rotation from the 2x2 block.
    transforms.push_back(
        {transform.at<double>(0, 2), transform.at<double>(1, 2),
         std::atan2(transform.at<double>(1, 0),
                    transform.at<double>(0, 0))});

    // Keep the diagnostic vector index-aligned with the transform vector.
    tracked_counts.push_back(tracked_count);

    // Roll the current grayscale image forward for the next pair.
    previous_gray = current_gray;
  }

  // A single-frame or unreadable clip has no motion sequence to stabilize.
  if (transforms.empty()) {
    throw std::runtime_error(
        "The input video must contain at least two frames.");
  }
  return transforms;
}

std::vector<Trajectory> cumulative_trajectory(
    const std::vector<Transform>& transforms) {
  // Preallocate one accumulated camera position for every pairwise transform.
  std::vector<Trajectory> trajectory;
  trajectory.reserve(transforms.size());

  // Start at the origin, then integrate translation and rotation over time.
  Trajectory accumulated;
  for (const Transform& transform : transforms) {
    accumulated.x += transform.dx;
    accumulated.y += transform.dy;
    accumulated.angle += transform.angle;
    trajectory.push_back(accumulated);
  }

  // This measured trajectory will be compared with its smoothed counterpart.
  return trajectory;
}

std::vector<Trajectory> smooth_trajectory(
    const std::vector<Trajectory>& trajectory, int radius) {
  // Preserve the measured trajectory and allocate a separate filtered result.
  std::vector<Trajectory> smoothed;
  smoothed.reserve(trajectory.size());

  // Produce one fixed-width moving-average sample at each trajectory position.
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

    // The clamped window always contains exactly 2 * radius + 1 samples.
    const double count = static_cast<double>(2 * radius + 1);
    smoothed.push_back(
        {sum.x / count, sum.y / count, sum.angle / count});
  }

  // Filtering axes independently keeps pixel translations separate from radians.
  return smoothed;
}

cv::Mat transform_matrix(const Transform& transform) {
  // Convert the stored angle back into the 2x2 rotation block.
  const double cosine = std::cos(transform.angle);
  const double sine = std::sin(transform.angle);
  // Matx provides fixed-size initialization without the deprecated Mat comma
  // initializer. Constructing Mat from it gives warpAffine the expected 2x3
  // double-precision matrix.
  const cv::Matx23d matrix(
      cosine, -sine, transform.dx, sine, cosine, transform.dy);
  return cv::Mat(matrix);
}

void fix_border(cv::Mat& frame) {
  // Scaling around the image center crops most empty wedges created by warping.
  const cv::Point2f center(
      static_cast<float>(frame.cols) / 2.0F,
      static_cast<float>(frame.rows) / 2.0F);
  const cv::Mat transform = cv::getRotationMatrix2D(center, 0.0, 1.04);

  // In-place warp keeps the original dimensions required by VideoWriter.
  cv::warpAffine(frame, frame, transform, frame.size());
}

int run(const Options& options) {
  // Construct and validate the destination before an encoder can truncate it.
  const std::filesystem::path output_path =
      options.output_dir / options.output_name;
  if (paths_refer_to_same_file(options.input, output_path)) {
    throw std::invalid_argument(
        "The input and output videos must use different paths.");
  }

  // VideoCapture chooses an available backend for the supplied path.
  cv::VideoCapture capture(options.input.string());
  if (!capture.isOpened()) {
    throw std::runtime_error(
        "Could not open input video: " + options.input.string());
  }

  // Read geometry and timing metadata before the two processing passes.
  const int width =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = capture.get(cv::CAP_PROP_FPS);

  // VideoWriter cannot be configured from missing or non-finite metadata.
  if (width <= 0 || height <= 0 || !std::isfinite(fps) || fps <= 0.0) {
    throw std::runtime_error(
        "The input video has invalid dimensions or frame rate.");
  }

  // Pass one estimates motion, integrates it, and smooths the camera trajectory.
  std::vector<std::size_t> tracked_counts;
  const std::vector<Transform> transforms =
      estimate_transforms(capture, tracked_counts);
  const std::vector<Trajectory> trajectory =
      cumulative_trajectory(transforms);
  const std::vector<Trajectory> smoothed =
      smooth_trajectory(trajectory, options.smoothing_radius);

  // The difference between smoothed and measured trajectories is the correction.
  std::vector<Transform> corrected;
  corrected.reserve(transforms.size());
  for (std::size_t index = 0; index < transforms.size(); ++index) {
    corrected.push_back(
        {transforms[index].dx + smoothed[index].x - trajectory[index].x,
         transforms[index].dy + smoothed[index].y - trajectory[index].y,
         transforms[index].angle + smoothed[index].angle -
             trajectory[index].angle});
  }

  // Reopen the path for pass two instead of relying on backend-specific seeking.
  capture.release();
  cv::VideoCapture render_capture(options.input.string());
  if (!render_capture.isOpened()) {
    throw std::runtime_error(
        "Could not reopen input video: " + options.input.string());
  }

  // Create the chosen destination explicitly before opening its video file.
  std::filesystem::create_directories(options.output_dir);

  // The tutorial output places original and stabilized frames side by side.
  int output_width = 2 * width;
  int output_height = height;

  // Halve very wide comparisons to keep preview and output sizes manageable.
  if (output_width > 1920) {
    output_width /= 2;
    output_height /= 2;
  }

  // mp4v is broadly available in common FFmpeg and AVFoundation OpenCV builds.
  cv::VideoWriter writer(
      output_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
      cv::Size(output_width, output_height));

  // Fail before pass two if the current backend cannot encode the destination.
  if (!writer.isOpened()) {
    render_capture.release();
    throw std::runtime_error(
        "Could not open output video: " + output_path.string());
  }

  // Apply one correction per adjacent frame pair in the reopened stream.
  std::size_t written_frames = 0;
  for (const Transform& transform : corrected) {
    cv::Mat frame;
    if (!render_capture.read(frame) || frame.empty()) {
      break;
    }

    // Warp away the estimated shake, then crop slightly to hide empty borders.
    cv::Mat stabilized;
    cv::warpAffine(
        frame, stabilized, transform_matrix(transform), frame.size());
    fix_border(stabilized);

    // Concatenate both views so learners can compare them frame by frame.
    cv::Mat comparison;
    cv::hconcat(frame, stabilized, comparison);

    // Wide inputs use the halved dimensions calculated for VideoWriter.
    if (comparison.cols != output_width ||
        comparison.rows != output_height) {
      cv::resize(
          comparison, comparison, cv::Size(output_width, output_height));
    }

    // Each encoded frame must match the writer's configured geometry.
    writer.write(comparison);
    ++written_frames;

    // Headless execution skips all GUI calls inside this conditional.
    if (options.display) {
      cv::imshow("Before and After", comparison);

      // Escape stops preview; --validate later rejects a partial run.
      if ((cv::waitKey(1) & 0xFF) == 27) {
        break;
      }
    }
  }

  // Flush the encoder and release native resources before inspecting output.
  render_capture.release();
  writer.release();
  if (options.display) {
    cv::destroyAllWindows();
  }

  // Optional validation checks semantics rather than merely a zero exit status.
  if (options.validate) {
    // A complete headless run writes one frame for every correction.
    if (written_frames != transforms.size()) {
      throw std::runtime_error(
          "The number of output frames did not match the transforms.");
    }

    // A nonempty regular file confirms that the encoder finalized its output.
    if (!std::filesystem::is_regular_file(output_path) ||
        std::filesystem::file_size(output_path) == 0U) {
      throw std::runtime_error("The output video is missing or empty.");
    }

    // Decode every result frame to catch a short or corrupt encoded stream.
    cv::VideoCapture check(output_path.string());
    std::size_t decoded_frames = 0;
    bool invalid_geometry = false;
    while (true) {
      cv::Mat output_frame;
      if (!check.read(output_frame) || output_frame.empty()) {
        break;
      }
      ++decoded_frames;
      if (output_frame.cols != output_width ||
          output_frame.rows != output_height) {
        invalid_geometry = true;
      }
    }
    check.release();
    if (decoded_frames == 0U) {
      throw std::runtime_error(
          "OpenCV could not decode the generated video.");
    }

    // Every decoded frame must match the dimensions supplied to VideoWriter.
    if (invalid_geometry) {
      throw std::runtime_error(
          "The generated video has unexpected dimensions.");
    }

    // Accepted writes are not enough: the finalized stream must contain all.
    if (decoded_frames != written_frames) {
      throw std::runtime_error(
          "The number of decoded output frames did not match the writes.");
    }

    // Mean successfully tracked corners provides a stable quality diagnostic.
    const double tracked_sum = std::accumulate(
        tracked_counts.begin(), tracked_counts.end(), 0.0);
    const double tracked_mean =
        tracked_sum / static_cast<double>(tracked_counts.size());
    std::cout << "VALIDATION PASSED: " << written_frames << " frames, "
              << output_width << 'x' << output_height
              << ", mean tracked points " << tracked_mean << '\n';
  }

  // Zero reports successful completion to shells and CTest.
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Keep parsing and reusable program logic separate from process-level errors.
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    // Expected input and runtime failures remain concise and actionable.
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
