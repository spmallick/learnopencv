// Command-line entry point for the LearnOpenCV optical-flow examples.

#include "optical_flow.hpp"

#include <opencv2/core/version.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using optical_flow::Algorithm;

struct CommandLine {
  Algorithm algorithm = Algorithm::kFarneback;
  optical_flow::RunOptions run_options{
      fs::path(OPTICAL_FLOW_PROJECT_DIR) / "videos" / "people.mp4",
      fs::path("outputs"),
      true,
      false,
      0};
  bool algorithm_was_set = false;
};

void print_usage(const char* executable) {
  std::cout
      << "Usage: " << executable << " --algorithm ALGORITHM [options]\n"
      << "Algorithms: farneback, lucaskanade, lucaskanade_dense, rlof\n"
      << "Options:\n"
      << "  --video PATH       Input video\n"
      << "  --output-dir PATH  Final visualization directory\n"
      << "  --max-frames N     Process N frame pairs; zero means all\n"
      << "  --no-display       Do not open GUI windows\n"
      << "  --validate         Check motion and output invariants\n";
}

Algorithm parse_algorithm(const std::string& value) {
  if (value == "farneback") {
    return Algorithm::kFarneback;
  }
  if (value == "lucaskanade") {
    return Algorithm::kLucasKanade;
  }
  if (value == "lucaskanade_dense") {
    return Algorithm::kLucasKanadeDense;
  }
  if (value == "rlof") {
    return Algorithm::kRlof;
  }
  throw std::invalid_argument("Unsupported algorithm: " + value);
}

CommandLine parse_command_line(int argc, char** argv) {
  CommandLine command_line;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--algorithm" && index + 1 < argc) {
      command_line.algorithm = parse_algorithm(argv[++index]);
      command_line.algorithm_was_set = true;
    } else if (argument == "--video" && index + 1 < argc) {
      command_line.run_options.video_path = argv[++index];
    } else if (argument == "--output-dir" && index + 1 < argc) {
      command_line.run_options.output_dir = argv[++index];
    } else if (argument == "--max-frames" && index + 1 < argc) {
      command_line.run_options.max_frames = std::stoi(argv[++index]);
    } else if (argument == "--no-display") {
      command_line.run_options.show_windows = false;
    } else if (argument == "--validate") {
      command_line.run_options.validate = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + argument);
    }
  }

  if (!command_line.algorithm_was_set) {
    throw std::invalid_argument("--algorithm is required");
  }
  if (command_line.run_options.max_frames < 0) {
    throw std::invalid_argument("--max-frames cannot be negative");
  }
  return command_line;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLine command_line = parse_command_line(argc, argv);
    const optical_flow::Summary summary =
        command_line.algorithm == Algorithm::kLucasKanade
        ? optical_flow::run_lucas_kanade(command_line.run_options)
        : optical_flow::run_dense_optical_flow(
              command_line.run_options,
              command_line.algorithm);

    std::cout << "OpenCV version: " << CV_VERSION << '\n'
              << "Algorithm: "
              << optical_flow::algorithm_name(command_line.algorithm) << '\n'
              << "Frame pairs processed: " << summary.frame_pairs << '\n'
              << "Mean motion magnitude: " << summary.mean_magnitude << '\n'
              << "Visualization: " << summary.output_path << '\n';
    if (command_line.run_options.validate) {
      std::cout << "VALIDATION PASSED: video, motion, and output checks\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
