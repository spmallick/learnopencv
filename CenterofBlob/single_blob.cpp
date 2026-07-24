// OpenCV core defines Mat, Point, Scalar, Moments, and countNonZero.
#include <opencv2/core.hpp>
// The version header exposes CV_VERSION and CV_VERSION_MAJOR.
#include <opencv2/core/version.hpp>
// highgui supplies the optional imshow, waitKey, and destroyAllWindows calls.
#include <opencv2/highgui.hpp>
// imgcodecs supplies imread and imwrite for the tracked PNG and generated output.
#include <opencv2/imgcodecs.hpp>
// imgproc supplies color conversion, thresholding, circles, and text rendering.
#include <opencv2/imgproc.hpp>

// OpenCV 5 moved moments and related shape functions into geometry.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

// cmath supplies std::abs for the zero-area guard.
#include <cmath>
// cstdlib supplies EXIT_SUCCESS and EXIT_FAILURE.
#include <cstdlib>
// exception supplies the common base type used by the CLI error boundary.
#include <exception>
// filesystem provides portable input and output paths in C++17.
#include <filesystem>
// iostream provides stable result output and clear error messages.
#include <iostream>
// optional represents the absence of an output destination without sentinels.
#include <optional>
// stdexcept provides invalid_argument and runtime_error.
#include <stdexcept>
// string stores command-line values before they are parsed.
#include <string>
// system_error lets filesystem operations report failures without ambiguity.
#include <system_error>

// Direct compiler invocations can still run by explicitly passing an input path.
#ifndef CENTER_OF_BLOB_SOURCE_DIR
#define CENTER_OF_BLOB_SOURCE_DIR "."
#endif

namespace {

// Use a short namespace alias for the many C++17 path operations below.
namespace fs = std::filesystem;

// CMake replaces this directory at compile time so the default works from any cwd.
const fs::path kDefaultInputPath =
    fs::path(CENTER_OF_BLOB_SOURCE_DIR) / "circle.png";
// Directory-based output always receives one predictable filename.
constexpr const char *kDefaultOutputName = "single_blob_result.png";
// These are the stable measurements of the tracked tutorial asset.
constexpr int kExpectedCenterX = 112;
constexpr int kExpectedCenterY = 112;
constexpr int kExpectedForegroundPixels = 35'272;

// Options records only parsed user intent and contains no OpenCV state.
struct Options {
  // The bundled asset is the default; --input or a positional path overrides it.
  fs::path image_path = kDefaultInputPath;
  // Track explicit input selection to reject two conflicting input spellings.
  bool input_was_set = false;
  // Threshold 127 separates the black circle from the white background.
  int threshold_value = 127;
  // The bundled example contains a dark object on a light background.
  bool foreground_dark = true;
  // --output writes to an exact caller-selected filename.
  std::optional<fs::path> output_path;
  // --output-dir writes kDefaultOutputName inside a caller-selected directory.
  std::optional<fs::path> output_directory;
  // Display is enabled for interactive teaching and disabled by --no-display.
  bool display = true;
  // --validate checks immutable sample metrics and emits a success marker.
  bool validate = false;
};

// Print both preferred switches and the legacy positional form.
void print_usage(const char *program) {
  // Keep the usage on one logical line so shell users can copy it easily.
  std::cout
      << "Usage: " << program
      << " [IMAGE | --input IMAGE] [--threshold 0..255]"
         " [--foreground dark|light] [--output FILE | --output-dir DIR]"
         " [--no-display] [--validate]\n";
}

// Parse the threshold strictly so values such as "12x" are not truncated.
int parse_threshold(const std::string &value) {
  // stoi reports how many input characters participated in the conversion.
  std::size_t parsed = 0;
  // Initialize before the try block to keep the range check straightforward.
  int result = 0;
  try {
    // Convert the complete decimal representation into an integer.
    result = std::stoi(value, &parsed);
  } catch (const std::exception &) {
    // Translate library-specific conversion failures into a stable CLI message.
    throw std::invalid_argument("--threshold expects an integer");
  }
  // Reject trailing characters and values outside an 8-bit intensity range.
  if (parsed != value.size() || result < 0 || result > 255) {
    throw std::invalid_argument("--threshold must be between 0 and 255");
  }
  // Return only a completely parsed, range-checked value.
  return result;
}

// Parse all supported spellings without reading files or invoking OpenCV.
Options parse_arguments(int argc, char **argv) {
  // Start with bundled-input and tutorial-friendly defaults.
  Options options;
  // Examine every token after the executable name exactly once.
  for (int index = 1; index < argc; ++index) {
    // Copy the current token so comparisons do not depend on raw pointers.
    const std::string argument = argv[index];
    // --help is a successful terminal action rather than an error.
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    // Automated tests must never wait for a window-system event.
    if (argument == "--no-display") {
      options.display = false;
      continue;
    }
    // Validation is independent of whether an output image is requested.
    if (argument == "--validate") {
      options.validate = true;
      continue;
    }
    // These switches each consume exactly one following token.
    if (argument == "--input" || argument == "-i" ||
        argument == "--ipimage" || argument == "--threshold" ||
        argument == "--foreground" || argument == "--output" ||
        argument == "--output-dir") {
      // Fail clearly rather than reading beyond argv when a value is absent.
      if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + argument);
      }
      // Advance once so the outer loop resumes after the consumed value.
      const std::string value = argv[++index];
      // All input aliases select the same image-path field.
      if (argument == "--input" || argument == "-i" ||
          argument == "--ipimage") {
        // Two inputs are ambiguous even when they happen to name the same file.
        if (options.input_was_set) {
          throw std::invalid_argument("Only one input image may be specified");
        }
        options.image_path = value;
        options.input_was_set = true;
      } else if (argument == "--threshold") {
        // Apply strict integer and range validation immediately.
        options.threshold_value = parse_threshold(value);
      } else if (argument == "--foreground") {
        // Only the two documented polarity words have defined semantics.
        if (value != "dark" && value != "light") {
          throw std::invalid_argument(
              "--foreground must be 'dark' or 'light'");
        }
        options.foreground_dark = value == "dark";
      } else if (argument == "--output") {
        // The two output modes would otherwise target competing filenames.
        if (options.output_directory.has_value()) {
          throw std::invalid_argument(
              "--output and --output-dir cannot be used together");
        }
        options.output_path = value;
      } else {
        // The remaining recognized value switch is --output-dir.
        if (options.output_path.has_value()) {
          throw std::invalid_argument(
              "--output and --output-dir cannot be used together");
        }
        options.output_directory = value;
      }
      continue;
    }
    // Any other dash-prefixed token is probably a misspelled option.
    if (!argument.empty() && argument.front() == '-') {
      throw std::invalid_argument("Unknown option: " + argument);
    }
    // Preserve the original tutorial's positional input-image syntax.
    if (options.input_was_set) {
      throw std::invalid_argument("Only one input image may be specified");
    }
    options.image_path = argument;
    options.input_was_set = true;
  }
  // Omitting an input intentionally leaves the bundled default in place.
  return options;
}

// Convert a BGR image into a mask whose intended foreground is nonzero.
cv::Mat create_binary_mask(const cv::Mat &image, int threshold_value,
                           bool foreground_dark) {
  // Convert to one channel because thresholding color channels independently
  // would make image moments difficult to interpret.
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  // Allocate the result through OpenCV so its size and type follow gray.
  cv::Mat binary_mask;
  // Invert dark objects so moments measure the object instead of its background.
  const int threshold_type =
      foreground_dark ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
  // Produce a strict 0/255 mask that is portable across OpenCV 4.14 and 5.
  cv::threshold(gray, binary_mask, threshold_value, 255, threshold_type);
  // Return the mask for both moment calculation and regression validation.
  return binary_mask;
}

// Compute the area-normalized first-order moment of the foreground mask.
cv::Point find_centroid(const cv::Mat &binary_mask) {
  // binaryImage=true gives every nonzero mask pixel equal weight.
  const cv::Moments image_moments = cv::moments(binary_mask, true);
  // m00 is zero when thresholding found no foreground pixels.
  if (std::abs(image_moments.m00) <= 1e-12) {
    throw std::runtime_error(
        "No foreground pixels were found in the binary mask");
  }
  // cvRound matches OpenCV's conventional conversion from subpixel centers.
  return {cvRound(image_moments.m10 / image_moments.m00),
          cvRound(image_moments.m01 / image_moments.m00)};
}

// Check all stable invariants of the repository's bundled circle image.
void validate_bundled_result(const fs::path &input_path, const cv::Mat &image,
                             const cv::Mat &binary_mask,
                             const cv::Point &centroid) {
  // equivalent compares filesystem identity and handles relative input spellings.
  std::error_code equivalence_error;
  const bool is_bundled =
      fs::equivalent(input_path, kDefaultInputPath, equivalence_error);
  // Never print a success marker for an arbitrary custom image.
  if (equivalence_error || !is_bundled) {
    throw std::runtime_error(
        "--validate requires the bundled circle.png input");
  }
  // Image dimensions and channel count verify successful decoding.
  if (image.rows != 225 || image.cols != 225 || image.channels() != 3) {
    throw std::runtime_error("Unexpected circle.png dimensions or channels");
  }
  // The centroid is the tutorial's central numerical result.
  if (centroid.x != kExpectedCenterX || centroid.y != kExpectedCenterY) {
    throw std::runtime_error("Bundled circle centroid did not match (112, 112)");
  }
  // The centered background shares this centroid, so also verify polarity using
  // the exact number of foreground pixels in the thresholded lossless PNG.
  if (cv::countNonZero(binary_mask) != kExpectedForegroundPixels) {
    throw std::runtime_error(
        "Bundled circle foreground-pixel count did not match 35272");
  }
}

// Resolve either output mode and create only its narrowly scoped parent directory.
std::optional<fs::path> resolve_output_path(const Options &options) {
  // An explicit output filename may include a new nested parent directory.
  if (options.output_path.has_value()) {
    // Avoid create_directories("") for a filename in the current directory.
    const fs::path parent = options.output_path->parent_path();
    if (!parent.empty()) {
      std::error_code create_error;
      fs::create_directories(parent, create_error);
      if (create_error) {
        throw std::runtime_error("Could not create output directory '" +
                                 parent.string() + "': " +
                                 create_error.message());
      }
    }
    return options.output_path;
  }
  // No output option means the example only prints and optionally displays.
  if (!options.output_directory.has_value()) {
    return std::nullopt;
  }
  // Create the requested directory before appending the deterministic filename.
  std::error_code create_error;
  fs::create_directories(*options.output_directory, create_error);
  if (create_error) {
    throw std::runtime_error("Could not create output directory '" +
                             options.output_directory->string() + "': " +
                             create_error.message());
  }
  return *options.output_directory / kDefaultOutputName;
}

// Encode the annotation and optionally prove the generated file is readable.
void write_annotated_image(const fs::path &output_path,
                           const cv::Mat &annotated, bool validate) {
  // imwrite returns false for encoder/write failures that do not throw.
  if (!cv::imwrite(output_path.string(), annotated)) {
    throw std::runtime_error("Could not write output image '" +
                             output_path.string() + "'");
  }
  // Validation checks the actual file rather than trusting only the return value.
  if (validate) {
    const cv::Mat decoded =
        cv::imread(output_path.string(), cv::IMREAD_COLOR);
    if (decoded.empty() || decoded.size() != annotated.size() ||
        decoded.type() != annotated.type()) {
      throw std::runtime_error("Could not verify output image '" +
                               output_path.string() + "'");
    }
  }
}

} // namespace

// The executable boundary converts all expected failures into a nonzero exit.
int main(int argc, char **argv) {
  try {
    // Parse CLI intent before performing image or filesystem operations.
    const Options options = parse_arguments(argc, argv);
    // Decode in color because annotation uses colored text and shapes.
    const cv::Mat image =
        cv::imread(options.image_path.string(), cv::IMREAD_COLOR);
    // A missing or unsupported input should not reach cvtColor.
    if (image.empty()) {
      std::cerr << "Error: could not read input image '"
                << options.image_path.string() << "'\n";
      return EXIT_FAILURE;
    }

    // Threshold with an explicit foreground polarity.
    const cv::Mat binary_mask = create_binary_mask(
        image, options.threshold_value, options.foreground_dark);
    // Measure the intended nonzero blob rather than the white background.
    const cv::Point centroid = find_centroid(binary_mask);
    // Clone before drawing so the decoded input remains unchanged.
    cv::Mat annotated = image.clone();
    // Mark the measured center with a filled white circle.
    cv::circle(annotated, centroid, 5, cv::Scalar(255, 255, 255),
               cv::FILLED);
    // Label the marker with APIs shared by OpenCV 4.14 and OpenCV 5.
    cv::putText(annotated, "centroid", centroid - cv::Point(25, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 2, cv::LINE_8);

    // Resolve output before validation so invalid destinations fail predictably.
    const std::optional<fs::path> output_path =
        resolve_output_path(options);
    // Verify the complete sample contract when regression mode is enabled.
    if (options.validate) {
      validate_bundled_result(options.image_path, image, binary_mask, centroid);
    }
    // Save only when the caller requested an output filename or directory.
    if (output_path.has_value()) {
      write_annotated_image(*output_path, annotated, options.validate);
    }

    // Print the stable primary metric in the same form as the Python example.
    std::cout << "Centroid: (" << centroid.x << ", " << centroid.y << ")\n";
    // Record the linked OpenCV version in every test log.
    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    // Emit a marker only after metrics and optional output verification pass.
    if (options.validate) {
      std::cout << "VALIDATION PASSED: single_blob\n";
    }
    // Keep display available for readers while allowing fully headless tests.
    if (options.display) {
      cv::imshow("Image with center", annotated);
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
  } catch (const std::exception &error) {
    // Present OpenCV, parsing, validation, and filesystem errors uniformly.
    std::cerr << "Error: " << error.what() << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  // Reaching this line means processing and every requested check succeeded.
  return EXIT_SUCCESS;
}
