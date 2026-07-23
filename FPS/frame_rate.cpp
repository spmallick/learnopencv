// OpenCV's core header provides cv::Mat, which stores each decoded frame.
#include <opencv2/core.hpp>
// The utility header exposes the linked library's runtime version string.
#include <opencv2/core/utility.hpp>
// The video I/O header provides VideoCapture and CAP_PROP_FPS.
#include <opencv2/videoio.hpp>

// algorithm provides std::max for scale-aware floating-point validation.
#include <algorithm>
// chrono provides the monotonic steady clock used for elapsed-time measurement.
#include <chrono>
// cmath provides finite-value and absolute-difference checks.
#include <cmath>
// exception provides the base type caught when integer parsing fails.
#include <exception>
// iomanip controls the precision of human-readable measurements.
#include <iomanip>
// iostream supplies stdout and stderr streams for results and diagnostics.
#include <iostream>
// stdexcept supplies distinct argument and runtime failure exceptions.
#include <stdexcept>
// string stores command-line options and video sources.
#include <string>

// Hide implementation helpers from other translation units.
namespace {

// Distinguish command-line misuse, which main reports with exit code 2.
class ArgumentError : public std::invalid_argument {
public:
  // Forward a clear parser diagnostic to std::invalid_argument.
  explicit ArgumentError(const std::string &message)
      : std::invalid_argument(message) {}
};

// Store every supported command-line choice with tutorial-compatible defaults.
struct Options {
  // A numeric string selects a camera; every other value is passed as a source path.
  std::string source = "0";
  // The original tutorial measured at most 120 frames.
  int num_frames = 120;
  // Warm-up reads are optional and excluded from timing.
  int warmup_frames = 0;
  // Validation is opt-in so normal output remains concise.
  bool validate = false;
  // Help is represented as data so parsing never bypasses stack unwinding.
  bool show_help = false;
};

// Keep the raw values used to calculate frame-read throughput together.
struct Measurement {
  // timed_fps is delivered read/decode throughput, not necessarily playback FPS.
  double timed_fps;
  // frames_read counts successful reads inside the timed section.
  int frames_read;
  // elapsed_seconds covers attempts, including a terminal failed EOF probe.
  double elapsed_seconds;
};

// Print one canonical usage string to either stdout or stderr.
void print_usage(std::ostream &stream, const char *program) {
  // Keep every public option visible in --help and parser-error output.
  stream << "Usage: " << program
         << " [--source CAMERA_OR_VIDEO] [--frames N]"
            " [--warmup-frames N] [--validate]\n";
}

// Parse a decimal CLI value and enforce a non-negative lower bound.
int parse_nonnegative_integer(const std::string &value,
                              const std::string &option) {
  // Track how many characters std::stoi consumed to reject values such as "5x".
  std::size_t parsed = 0;
  // Initialize the result before entering the exception-producing conversion.
  int result = 0;
  // Translate standard conversion errors into option-specific diagnostics.
  try {
    // std::stoi also detects values outside the range of int.
    result = std::stoi(value, &parsed);
  // invalid_argument and out_of_range both derive from std::exception.
  } catch (const std::exception &) {
    // Hide implementation-specific stoi wording from the tutorial user.
    throw ArgumentError(option + " expects an integer");
  }
  // Require a complete conversion and disallow negative counts.
  if (parsed != value.size() || result < 0) {
    // State the accepted domain in the diagnostic.
    throw ArgumentError(option + " expects a non-negative integer");
  }
  // Return a checked value that is safe to use as a loop bound.
  return result;
}

// Identify options that require exactly one following value.
bool option_requires_value(const std::string &argument) {
  // Only source and the two frame-count controls consume another token.
  return argument == "--source" || argument == "--frames" ||
         argument == "--warmup-frames";
}

// Convert argv into Options without opening a camera or file.
Options parse_arguments(int argc, char **argv) {
  // Begin with the tutorial's documented default values.
  Options options;
  // Visit each user-supplied token once.
  for (int index = 1; index < argc; ++index) {
    // Copy the current token so comparisons and diagnostics are straightforward.
    const std::string argument = argv[index];
    // Help is a flag and therefore consumes no value.
    if (argument == "--help" || argument == "-h") {
      // Let main print help and return normally with status zero.
      options.show_help = true;
      // Match argparse by ending parsing immediately once help is requested.
      return options;
    }
    // Validation is also a value-free flag.
    if (argument == "--validate") {
      // Record the request for intrinsic post-measurement checks.
      options.validate = true;
      // Move directly to the next token.
      continue;
    }
    // Reject unknown tokens before looking for a value, producing an accurate error.
    if (!option_requires_value(argument)) {
      // Include the exact unrecognized token in the diagnostic.
      throw ArgumentError("Unknown option: " + argument);
    }
    // A recognized value-taking option cannot be the final token.
    if (index + 1 >= argc) {
      // Report the option whose value is absent.
      throw ArgumentError("Missing value for " + argument);
    }
    // Consume the value exactly once.
    const std::string value = argv[++index];
    // Treat another long option as a missing value rather than as a filename.
    if (value.rfind("--", 0) == 0) {
      // This avoids the misleading "--source --frames" interpretation.
      throw ArgumentError("Missing value for " + argument);
    }
    // Store the source verbatim until its camera/file interpretation is needed.
    if (argument == "--source") {
      // Numeric-looking sources are converted later by open_video_source().
      options.source = value;
    // Parse and validate the maximum number of timed reads.
    } else if (argument == "--frames") {
      // Reuse the common integer parser for consistent diagnostics.
      options.num_frames = parse_nonnegative_integer(value, argument);
      // Zero cannot produce a meaningful timed FPS measurement.
      if (options.num_frames == 0) {
        // Keep the more precise semantic error instead of saying non-negative.
        throw ArgumentError("--frames must be greater than zero");
      }
    // The only remaining recognized value option is --warmup-frames.
    } else {
      // Zero is allowed because warm-up is optional.
      options.warmup_frames = parse_nonnegative_integer(value, argument);
    }
  }
  // Return the fully validated command-line configuration.
  return options;
}

// Recognize a signed decimal camera index that fits OpenCV's int overload.
bool parse_camera_index(const std::string &source, int &camera_index) {
  // Track complete consumption so a path such as "12.mp4" stays a filename.
  std::size_t parsed = 0;
  // Attempt the same bounded integer conversion used for count options.
  try {
    // Write the converted index through the reference only when parsing begins.
    camera_index = std::stoi(source, &parsed);
  // Any conversion failure means the source should be handled as a string.
  } catch (const std::exception &) {
    // Signal that VideoCapture's string overload should be selected.
    return false;
  }
  // A partial conversion is a filename or URL, not a camera index.
  return parsed == source.size();
}

// Select the correct VideoCapture constructor from the source's textual form.
cv::VideoCapture open_video_source(const std::string &source) {
  // Reserve storage for a possible camera index.
  int camera_index = 0;
  // A complete integer selects the camera-device overload.
  if (parse_camera_index(source, camera_index)) {
    // OpenCV interprets the integer according to the active camera backend.
    return cv::VideoCapture(camera_index);
  }
  // All other strings are delegated to OpenCV as files, URLs, or image sequences.
  return cv::VideoCapture(source);
}

// Normalize version-specific unavailable-property sentinels to one stable value.
double get_reported_fps(const cv::VideoCapture &video) {
  // Ask the active backend for its reported frame-rate property value.
  const double fps = video.get(cv::CAP_PROP_FPS);
  // OpenCV 4 commonly uses 0 and OpenCV 5 uses -1 when a property is unavailable.
  if (std::isfinite(fps) && fps > 0.0) {
    // Preserve valid fractional reported values such as 29.97.
    return fps;
  }
  // Normalize non-positive and non-finite backend results for dual-major behavior.
  return 0.0;
}

// Time a bounded read-attempt loop using a monotonic clock.
Measurement measure_fps(cv::VideoCapture &video, int num_frames,
                        int warmup_frames) {
  // Reuse one Mat so OpenCV can manage frame-buffer allocation efficiently.
  cv::Mat frame;
  // Advance past optional setup frames before starting the timer.
  for (int index = 0; index < warmup_frames; ++index) {
    // read() combines frame grabbing and decoding.
    if (!video.read(frame)) {
      // Explain that the source ended before the timed section was reached.
      throw std::runtime_error("Could not read a frame during warm-up");
    }
  }

  // Start the monotonic timer immediately before the first measured read.
  const auto start = std::chrono::steady_clock::now();
  // Count only frames whose read() operation succeeds.
  int frames_read = 0;
  // Bound camera measurements while allowing short files to end early.
  for (int index = 0; index < num_frames; ++index) {
    // Stop when a file ends, a camera disconnects, or the backend reports failure.
    if (!video.read(frame)) {
      // Do not count the failed read.
      break;
    }
    // Record exactly one successfully decoded frame.
    ++frames_read;
  }
  // Stop after the final attempt, including the failed probe that detects EOF.
  const auto end = std::chrono::steady_clock::now();
  // Convert the clock's native duration into portable fractional seconds.
  const double elapsed = std::chrono::duration<double>(end - start).count();

  // A successful FPS sample requires at least one returned frame.
  if (frames_read == 0) {
    // Avoid returning a misleading zero or dividing an empty sample.
    throw std::runtime_error("Could not read any frames from the video source");
  }
  // Protect the divisor from invalid or theoretically impossible clock results.
  if (!std::isfinite(elapsed) || elapsed <= 0.0) {
    // Keep failure explicit if a platform clock violates the expected invariant.
    throw std::runtime_error(
        "The elapsed time must be finite and greater than zero");
  }
  // Compute read throughput from successful frames rather than the requested count.
  const double timed_fps = frames_read / elapsed;
  // Reject overflow or any non-positive rate before it reaches output.
  if (!std::isfinite(timed_fps) || timed_fps <= 0.0) {
    // A positive frame count and duration must produce a positive finite rate.
    throw std::runtime_error(
        "The timed frame-read rate must be finite and positive");
  }
  // Return the raw values used by printing and optional validation.
  return {timed_fps, frames_read, elapsed};
}

// Check machine-independent relationships without imposing a speed threshold.
void validate_measurement(double reported_fps, const Measurement &measurement,
                          int requested_frames) {
  // The normalized backend value must be finite and non-negative.
  if (!std::isfinite(reported_fps) || reported_fps < 0.0) {
    // Detect regressions in unavailable-property handling.
    throw std::runtime_error("Reported FPS validation failed");
  }
  // The successful sample cannot be empty or exceed its configured maximum.
  if (measurement.frames_read < 1 ||
      measurement.frames_read > requested_frames) {
    // Detect both frame overcounting and an impossible empty success.
    throw std::runtime_error("Frame-count validation failed");
  }
  // Preserve the finite positive duration invariant.
  if (!std::isfinite(measurement.elapsed_seconds) ||
      measurement.elapsed_seconds <= 0.0) {
    // Keep this function safe for measurements constructed outside measure_fps().
    throw std::runtime_error("Elapsed-time validation failed");
  }
  // Preserve the finite positive throughput invariant.
  if (!std::isfinite(measurement.timed_fps) ||
      measurement.timed_fps <= 0.0) {
    // Reject a corrupted or externally constructed result.
    throw std::runtime_error("Timed-rate validation failed");
  }

  // Recompute the rate from its defining raw values.
  const double expected_rate =
      measurement.frames_read / measurement.elapsed_seconds;
  // Scale a tight tolerance to the magnitude of the calculated rate.
  const double tolerance = std::max(1.0, std::abs(expected_rate)) * 1e-12;
  // Permit normal floating-point rounding but no meaningful inconsistency.
  if (std::abs(measurement.timed_fps - expected_rate) > tolerance) {
    // Ensure displayed throughput always describes the measured sample.
    throw std::runtime_error("Timed-rate calculation validation failed");
  }
}

// End the file-local implementation namespace.
} // namespace

// Coordinate parsing, capture, reporting, validation, and process exit codes.
int main(int argc, char **argv) {
  // Parse separately so argument misuse can return code 2 like Python argparse.
  Options options;
  // Convert argv without touching a camera or file.
  try {
    // Store the checked configuration for the runtime phase.
    options = parse_arguments(argc, argv);
  // ArgumentError represents only command-line misuse.
  } catch (const ArgumentError &error) {
    // Print a concise diagnostic before the usage reminder.
    std::cerr << "Error: " << error.what() << '\n';
    // Put usage on stderr because this is an unsuccessful invocation.
    print_usage(std::cerr, argv[0]);
    // Match the conventional argparse status for invalid syntax or values.
    return 2;
  }

  // Help is a successful request and should not open the default camera.
  if (options.show_help) {
    // Put requested help on stdout for shell piping.
    print_usage(std::cout, argv[0]);
    // Return success without bypassing automatic object destruction.
    return 0;
  }

  // Convert runtime and validation exceptions into exit code 1.
  try {
    // Select the camera-index or string-source VideoCapture constructor.
    cv::VideoCapture video = open_video_source(options.source);
    // Check backend initialization before querying metadata or reading frames.
    if (!video.isOpened()) {
      // Include the exact source in the diagnostic.
      throw std::runtime_error("Could not open video source '" + options.source +
                               "'");
    }

    // Query the backend-reported property before reads advance the stream position.
    const double reported_fps = get_reported_fps(video);
    // Perform warm-up and timed reads with the opened source.
    const Measurement result =
        measure_fps(video, options.num_frames, options.warmup_frames);
    // Run optional checks against unrounded raw values.
    if (options.validate) {
      // Avoid platform-specific assertions about actual decode or camera speed.
      validate_measurement(reported_fps, result, options.num_frames);
    }

    // Release the backend promptly instead of waiting for main() to return.
    video.release();
    // Use fixed formatting for stable, machine-parseable matrix logs.
    std::cout << std::fixed;
    // Report the version exported by the linked OpenCV core library at runtime.
    std::cout << "OpenCV version: " << cv::getVersionString() << '\n';
    // Print the normalized backend value separately from timed throughput.
    std::cout << "Reported frames per second: " << std::setprecision(3)
              << reported_fps << '\n';
    // Echo the configured maximum number of timed reads.
    std::cout << "Frames requested: " << options.num_frames << '\n';
    // Echo discarded setup reads for short-file interpretation.
    std::cout << "Warm-up frames: " << options.warmup_frames << '\n';
    // Report actual successful reads so early end-of-file is visible.
    std::cout << "Frames read: " << result.frames_read << '\n';
    // Preserve enough precision that fast file decoding does not display as zero.
    std::cout << "Time taken: " << std::setprecision(9)
              << result.elapsed_seconds << " seconds\n";
    // Label throughput carefully so it is not confused with playback FPS.
    std::cout << "Timed frame-read rate: " << std::setprecision(3)
              << result.timed_fps << " frames/second\n";
    // Emit a stable marker only after every requested validation check passed.
    if (options.validate) {
      // CTest and manual runs use this exact text as their success criterion.
      std::cout << "Validation passed\n";
    }
  // Capture and validation failures are runtime errors rather than CLI misuse.
  } catch (const std::exception &error) {
    // Send runtime diagnostics to stderr so stdout contains only valid results.
    std::cerr << "Error: " << error.what() << '\n';
    // Return a general runtime-failure status.
    return 1;
  }
  // Signal successful measurement and optional validation.
  return 0;
}
