// OpenCV core defines Mat, Point, Scalar, Moments, and countNonZero.
#include <opencv2/core.hpp>
// The version header exposes CV_VERSION and CV_VERSION_MAJOR.
#include <opencv2/core/version.hpp>
// highgui supplies optional window display and keyboard waiting.
#include <opencv2/highgui.hpp>
// imgcodecs loads the bundled PNG and writes the annotated result.
#include <opencv2/imgcodecs.hpp>
// imgproc supplies thresholding, contour extraction, and annotation drawing.
#include <opencv2/imgproc.hpp>

// OpenCV 5 moved contourArea and moments into the geometry module.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

// algorithm supplies the deterministic geometric sort.
#include <algorithm>
// array stores the fixed bundled-image validation metrics.
#include <array>
// cmath supplies finite-number and zero-area checks.
#include <cmath>
// cstdlib supplies conventional process exit codes.
#include <cstdlib>
// exception supplies the common CLI error boundary.
#include <exception>
// filesystem provides portable C++17 paths and directory creation.
#include <filesystem>
// iomanip formats contour areas with one stable decimal place.
#include <iomanip>
// iostream provides result output and diagnostic errors.
#include <iostream>
// optional represents the absence of an output destination.
#include <optional>
// stdexcept supplies invalid_argument and runtime_error.
#include <stdexcept>
// string stores command-line tokens and labels.
#include <string>
// system_error reports filesystem failures without throwing prematurely.
#include <system_error>
// vector stores contours and the variable-length collection of blobs.
#include <vector>

// Direct compiler invocations can still run when they pass an explicit image.
#ifndef CENTER_OF_BLOB_SOURCE_DIR
#define CENTER_OF_BLOB_SOURCE_DIR "."
#endif

namespace {

// Shorten the repeated C++17 filesystem namespace.
namespace fs = std::filesystem;

// CMake embeds the tracked source directory for cwd-independent default input.
const fs::path kDefaultInputPath =
    fs::path(CENTER_OF_BLOB_SOURCE_DIR) / "multiple-blob.png";
// --output-dir writes exactly this one deterministic artifact.
constexpr const char *kDefaultOutputName = "multiple_blob_result.png";
// The lossless tutorial image produces these left-to-right centroids.
const std::array<cv::Point, 6> kExpectedCentroids = {
    cv::Point(72, 130), cv::Point(256, 115), cv::Point(439, 115),
    cv::Point(625, 115), cv::Point(807, 122), cv::Point(993, 116)};
// These corresponding contour areas are stable across the tested majors.
constexpr std::array<double, 6> kExpectedAreas = {
    10'194.0, 13'946.0, 14'583.0, 16'105.0, 3'688.0, 14'853.0};
// The thresholded sample contains this many foreground pixels.
constexpr int kExpectedForegroundPixels = 70'791;

// Options contains parsed CLI values but no decoded or derived image state.
struct Options {
  // The bundled sample is used when no override is provided.
  fs::path image_path = kDefaultInputPath;
  // Track whether an input alias or positional path already selected an image.
  bool input_was_set = false;
  // Threshold 127 separates all six shapes from the white background.
  int threshold_value = 127;
  // The sample's foreground shapes are darker than their background.
  bool foreground_dark = true;
  // Ignore only degenerate contours by default.
  double min_area = 1.0;
  // --output selects an exact filename.
  std::optional<fs::path> output_path;
  // --output-dir selects a directory for kDefaultOutputName.
  std::optional<fs::path> output_directory;
  // Interactive display is on unless explicitly disabled.
  bool display = true;
  // Validation checks sample metrics and prints an unambiguous marker.
  bool validate = false;
};

// Blob keeps the geometry required for drawing, sorting, and reporting together.
struct Blob {
  // The extracted external boundary is retained for drawContours.
  std::vector<cv::Point> contour;
  // The centroid is the area-normalized first-order contour moment.
  cv::Point centroid;
  // The geometric contour area supports filtering and regression checks.
  double area;
};

// Describe preferred options while retaining the original positional syntax.
void print_usage(const char *program) {
  // One logical line remains straightforward to copy into a terminal.
  std::cout
      << "Usage: " << program
      << " [IMAGE | --input IMAGE] [--threshold 0..255]"
         " [--foreground dark|light] [--min-area N]"
         " [--output FILE | --output-dir DIR] [--no-display] [--validate]\n";
}

// Convert a threshold token and require every character to be valid.
int parse_threshold(const std::string &value) {
  // stoi writes the number of characters it consumed into parsed.
  std::size_t parsed = 0;
  // Initialize outside the try block for a single range-check path.
  int result = 0;
  try {
    // Parse a base-10 signed integer.
    result = std::stoi(value, &parsed);
  } catch (const std::exception &) {
    // Hide platform-specific conversion wording behind a stable message.
    throw std::invalid_argument("--threshold expects an integer");
  }
  // Reject trailing text and values outside the 8-bit image range.
  if (parsed != value.size() || result < 0 || result > 255) {
    throw std::invalid_argument("--threshold must be between 0 and 255");
  }
  // Return only a completely parsed, valid threshold.
  return result;
}

// Convert and validate the minimum geometric contour area.
double parse_min_area(const std::string &value) {
  // stod reports how many source characters were consumed.
  std::size_t parsed = 0;
  // Initialize once so conversion and semantic checks share the result.
  double result = 0.0;
  try {
    // Parse decimal and scientific-notation values accepted by std::stod.
    result = std::stod(value, &parsed);
  } catch (const std::exception &) {
    // Present one predictable diagnostic for invalid numeric text.
    throw std::invalid_argument("--min-area expects a number");
  }
  // NaN, infinity, negatives, and trailing characters are all meaningless here.
  if (parsed != value.size() || !std::isfinite(result) || result < 0.0) {
    throw std::invalid_argument(
        "--min-area must be a finite non-negative number");
  }
  // Return the checked filtering threshold.
  return result;
}

// Parse arguments without reading images or creating output directories.
Options parse_arguments(int argc, char **argv) {
  // Begin with the bundled sample and tutorial-friendly defaults.
  Options options;
  // Inspect each token after the executable name.
  for (int index = 1; index < argc; ++index) {
    // Copy the raw token into a safe standard string.
    const std::string argument = argv[index];
    // Help is a successful request and should not enter the error handler.
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    // Headless runs skip every window-system call.
    if (argument == "--no-display") {
      options.display = false;
      continue;
    }
    // Regression mode may be combined with either output mode.
    if (argument == "--validate") {
      options.validate = true;
      continue;
    }
    // Each recognized value switch must consume one following token.
    if (argument == "--input" || argument == "-i" ||
        argument == "--ipimage" || argument == "--threshold" ||
        argument == "--foreground" || argument == "--min-area" ||
        argument == "--output" || argument == "--output-dir") {
      // Refuse an incomplete option before indexing past argv.
      if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + argument);
      }
      // Consume the value now so the loop advances to the next option.
      const std::string value = argv[++index];
      // All three input spellings update the same field.
      if (argument == "--input" || argument == "-i" ||
          argument == "--ipimage") {
        // Multiple input declarations are ambiguous and likely accidental.
        if (options.input_was_set) {
          throw std::invalid_argument("Only one input image may be specified");
        }
        options.image_path = value;
        options.input_was_set = true;
      } else if (argument == "--threshold") {
        // Apply strict 8-bit range validation immediately.
        options.threshold_value = parse_threshold(value);
      } else if (argument == "--foreground") {
        // Only the two documented mask polarities are meaningful.
        if (value != "dark" && value != "light") {
          throw std::invalid_argument(
              "--foreground must be 'dark' or 'light'");
        }
        options.foreground_dark = value == "dark";
      } else if (argument == "--min-area") {
        // Reject non-finite and negative filters during parsing.
        options.min_area = parse_min_area(value);
      } else if (argument == "--output") {
        // One invocation cannot choose competing output naming strategies.
        if (options.output_directory.has_value()) {
          throw std::invalid_argument(
              "--output and --output-dir cannot be used together");
        }
        options.output_path = value;
      } else {
        // The only remaining recognized value option is --output-dir.
        if (options.output_path.has_value()) {
          throw std::invalid_argument(
              "--output and --output-dir cannot be used together");
        }
        options.output_directory = value;
      }
      continue;
    }
    // Treat unrecognized dash-prefixed text as an option typo.
    if (!argument.empty() && argument.front() == '-') {
      throw std::invalid_argument("Unknown option: " + argument);
    }
    // Keep the article's original positional image argument working.
    if (options.input_was_set) {
      throw std::invalid_argument("Only one input image may be specified");
    }
    options.image_path = argument;
    options.input_was_set = true;
  }
  // No explicit input intentionally leaves the bundled default selected.
  return options;
}

// Convert a color image into a binary mask with the intended blobs in white.
cv::Mat create_binary_mask(const cv::Mat &image, int threshold_value,
                           bool foreground_dark) {
  // Contours operate on a single-channel image, so first discard color.
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  // OpenCV allocates the threshold result based on gray's size and depth.
  cv::Mat binary_mask;
  // Invert dark shapes so contour extraction sees them as foreground.
  const int threshold_type =
      foreground_dark ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
  // Produce a strict 0/255 mask shared by OpenCV 4.14 and 5.
  cv::threshold(gray, binary_mask, threshold_value, 255, threshold_type);
  // Preserve the mask for both contours and independent validation metrics.
  return binary_mask;
}

// Extract valid external blobs and impose a stable geometric order.
std::vector<Blob> find_blobs(const cv::Mat &binary_mask, double min_area) {
  // findContours fills a nested vector with the points on each boundary.
  std::vector<std::vector<cv::Point>> contours;
  // Pass a clone to avoid relying on version-specific input mutation behavior;
  // RETR_EXTERNAL intentionally ignores nested edge pairs and holes.
  cv::findContours(binary_mask.clone(), contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Reserve the maximum possible result size to avoid repeated reallocations.
  std::vector<Blob> blobs;
  blobs.reserve(contours.size());
  // Examine every candidate boundary returned by OpenCV.
  for (const auto &contour : contours) {
    // contourArea measures the area enclosed by this boundary.
    const double area = cv::contourArea(contour);
    // Drop tiny regions before computing their more expensive moments.
    if (area < min_area) {
      continue;
    }
    // Contour moments contain both signed area and first-order coordinate sums.
    const cv::Moments contour_moments = cv::moments(contour, false);
    // Lines and isolated points have no enclosed area and no defined centroid.
    if (std::abs(contour_moments.m00) <= 1e-12) {
      continue;
    }
    // Normalize first-order moments and use OpenCV's conventional rounding.
    const cv::Point centroid(
        cvRound(contour_moments.m10 / contour_moments.m00),
        cvRound(contour_moments.m01 / contour_moments.m00));
    // Retain all information required by output, drawing, and validation.
    blobs.push_back({contour, centroid, area});
  }
  // The API does not guarantee contour vector order, so sort by stable geometry.
  std::sort(blobs.begin(), blobs.end(),
            [](const Blob &left, const Blob &right) {
              // Horizontal position is the tutorial's primary reading order.
              if (left.centroid.x != right.centroid.x) {
                return left.centroid.x < right.centroid.x;
              }
              // Vertical position resolves blobs sharing the same x coordinate.
              if (left.centroid.y != right.centroid.y) {
                return left.centroid.y < right.centroid.y;
              }
              // Area provides a final deterministic tie-breaker.
              return left.area < right.area;
            });
  // Return a version-independent sequence for annotation and comparison.
  return blobs;
}

// Check dimensions, mask polarity, centroids, and areas for the bundled asset.
void validate_bundled_result(const fs::path &input_path, const cv::Mat &image,
                             const cv::Mat &binary_mask,
                             const std::vector<Blob> &blobs) {
  // Filesystem identity accepts absolute, relative, and symlinked spellings.
  std::error_code equivalence_error;
  const bool is_bundled =
      fs::equivalent(input_path, kDefaultInputPath, equivalence_error);
  // Never claim regression success for an arbitrary user image.
  if (equivalence_error || !is_bundled) {
    throw std::runtime_error(
        "--validate requires the bundled multiple-blob.png input");
  }
  // Confirm decoding produced the known width, height, and BGR channels.
  if (image.rows != 236 || image.cols != 1089 || image.channels() != 3) {
    throw std::runtime_error(
        "Unexpected multiple-blob.png dimensions or channels");
  }
  // Exactly six external foreground shapes are part of the sample contract.
  if (blobs.size() != kExpectedCentroids.size()) {
    throw std::runtime_error("Bundled image did not produce exactly six blobs");
  }
  // Check each sorted centroid and its corresponding contour area.
  for (std::size_t index = 0; index < blobs.size(); ++index) {
    // Point equality compares both integer coordinates.
    if (blobs[index].centroid != kExpectedCentroids[index]) {
      throw std::runtime_error(
          "Bundled blob centroids did not match expected values");
    }
    // The lossless input produces exact integer-valued geometric areas.
    if (std::abs(blobs[index].area - kExpectedAreas[index]) > 1e-6) {
      throw std::runtime_error(
          "Bundled blob contour areas did not match expected values");
    }
  }
  // Mask size catches a polarity/threshold regression independently of contours.
  if (cv::countNonZero(binary_mask) != kExpectedForegroundPixels) {
    throw std::runtime_error(
        "Bundled image foreground-pixel count did not match 70791");
  }
}

// Create only the requested output location and return its final filename.
std::optional<fs::path> resolve_output_path(const Options &options) {
  // --output may include a nested parent directory.
  if (options.output_path.has_value()) {
    // A plain filename has an empty parent and needs no mkdir operation.
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
  // With neither output switch, processing has no filesystem side effect.
  if (!options.output_directory.has_value()) {
    return std::nullopt;
  }
  // Create the caller-selected directory before appending the standard name.
  std::error_code create_error;
  fs::create_directories(*options.output_directory, create_error);
  if (create_error) {
    throw std::runtime_error("Could not create output directory '" +
                             options.output_directory->string() + "': " +
                             create_error.message());
  }
  return *options.output_directory / kDefaultOutputName;
}

// Write the annotation and, in validation mode, decode the real artifact.
void write_annotated_image(const fs::path &output_path,
                           const cv::Mat &annotated, bool validate) {
  // A false return means the encoder or filesystem could not complete the write.
  if (!cv::imwrite(output_path.string(), annotated)) {
    throw std::runtime_error("Could not write output image '" +
                             output_path.string() + "'");
  }
  // Reading back catches truncated, inaccessible, or wrong-format output.
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

// Convert all expected parse, OpenCV, validation, and filesystem errors to exits.
int main(int argc, char **argv) {
  try {
    // Parse every option before opening the input or creating output.
    const Options options = parse_arguments(argc, argv);
    // Decode in color because the final contours and labels are colored.
    const cv::Mat image =
        cv::imread(options.image_path.string(), cv::IMREAD_COLOR);
    // Avoid passing an empty matrix into cvtColor.
    if (image.empty()) {
      std::cerr << "Error: could not read input image '"
                << options.image_path.string() << "'\n";
      return EXIT_FAILURE;
    }

    // Convert the selected foreground polarity into a white-on-black mask.
    const cv::Mat binary_mask = create_binary_mask(
        image, options.threshold_value, options.foreground_dark);
    // Extract external shapes, filter degenerate regions, and sort geometrically.
    const std::vector<Blob> blobs =
        find_blobs(binary_mask, options.min_area);
    // A fully filtered result is a settings/input error, not successful output.
    if (blobs.empty()) {
      throw std::runtime_error(
          "No blobs matched the threshold and area settings");
    }

    // Clone before annotation so validation retains the original decoded pixels.
    cv::Mat annotated = image.clone();
    // Print all areas with the same single decimal place as Python.
    std::cout << std::fixed << std::setprecision(1);
    // Draw and report blobs in deterministic left-to-right order.
    for (std::size_t index = 0; index < blobs.size(); ++index) {
      // Give this iteration a readable reference to its measured geometry.
      const Blob &blob = blobs[index];
      // drawContours accepts a nested vector, so wrap this one retained boundary.
      cv::drawContours(
          annotated, std::vector<std::vector<cv::Point>>{blob.contour}, -1,
          cv::Scalar(167, 151, 0), 2);
      // Mark the area-normalized centroid with a filled white circle.
      cv::circle(annotated, blob.centroid, 5, cv::Scalar(255, 255, 255),
                 cv::FILLED);
      // Number the image in the same one-based order printed below.
      cv::putText(annotated, std::to_string(index + 1),
                  blob.centroid + cv::Point(8, -8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 255, 255), 2, cv::LINE_8);
      // Emit exact geometry for cross-version and cross-language comparison.
      std::cout << "Blob " << index + 1 << ": centroid=(" << blob.centroid.x
                << ", " << blob.centroid.y << "), area=" << blob.area << '\n';
    }

    // Resolve and create the optional output destination.
    const std::optional<fs::path> output_path =
        resolve_output_path(options);
    // Validate stable sample metrics before printing a success marker.
    if (options.validate) {
      validate_bundled_result(options.image_path, image, binary_mask, blobs);
    }
    // Encode output only when the caller requested a file or directory.
    if (output_path.has_value()) {
      write_annotated_image(*output_path, annotated, options.validate);
    }

    // Record the exact linked runtime in every acceptance log.
    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    // Emit the marker only after optional output verification has succeeded.
    if (options.validate) {
      std::cout << "VALIDATION PASSED: center_of_multiple_blob\n";
    }
    // Preserve the interactive tutorial experience outside headless runs.
    if (options.display) {
      cv::imshow("Blob centroids", annotated);
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
  } catch (const std::exception &error) {
    // Standardize parse, OpenCV, validation, and filesystem diagnostics.
    std::cerr << "Error: " << error.what() << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  // Reaching here means every requested operation succeeded.
  return EXIT_SUCCESS;
}
