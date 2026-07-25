#pragma once

#include <filesystem>
#include <string>

namespace optical_flow {

enum class Algorithm {
  kFarneback,
  kLucasKanade,
  kLucasKanadeDense,
  kRlof,
};

struct RunOptions {
  std::filesystem::path video_path;
  std::filesystem::path output_dir;
  bool show_windows = true;
  bool validate = false;
  int max_frames = 0;
};

struct Summary {
  int frame_pairs = 0;
  double mean_magnitude = 0.0;
  std::filesystem::path output_path;
};

Summary run_lucas_kanade(const RunOptions& options);

Summary run_dense_optical_flow(
    const RunOptions& options,
    Algorithm algorithm);

std::string algorithm_name(Algorithm algorithm);

}  // namespace optical_flow
