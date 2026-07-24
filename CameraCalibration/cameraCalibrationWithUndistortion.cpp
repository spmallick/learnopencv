// Shared helpers keep both tutorial executables numerically identical.
#include "calibration_utils.hpp"

// Filesystem resolves the bundled wildcard and explicit output directory.
#include <filesystem>
// The CLI prints help, results, validation markers, and errors.
#include <iostream>
// Invalid options and failed invariants use standard exceptions.
#include <stdexcept>
// Strings store option tokens and the calibration image pattern.
#include <string>

// Keep path-heavy option handling concise.
namespace fs = std::filesystem;

// Both exact reference versions remain below this mean intensity difference.
constexpr double kMaximumMethodDifference = 0.1;

// Hold command-line choices separately from calibration and image results.
struct Options {
    // CMake supplies an absolute bundled-asset directory at compile time.
    std::string images =
        (fs::path(TUTORIAL_IMAGE_DIR) / "*.jpg").string();
    // An empty sample means "use the first sorted calibration image."
    fs::path sample;
    // Generated images go to the caller's current directory by default.
    fs::path output_dir = fs::current_path();
    // Preserve the original interactive lesson unless headless mode is selected.
    bool display = true;
    // Validation remains optional for custom datasets.
    bool validate = false;
    // Help exits without reading images or creating output.
    bool show_help = false;
};

// Print the exact interface accepted by this executable.
void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " [--images GLOB] [--sample IMAGE] [--output-dir PATH]\n"
        << "       [--no-display] [--validate]\n"
        << "\n"
        << "Options:\n"
        << "  --images GLOB     Calibration image pattern (default: bundled images)\n"
        << "  --sample IMAGE    Image to undistort (default: first calibration image)\n"
        << "  --output-dir PATH Output directory (default: current directory)\n"
        << "  --no-display      Skip all interactive windows\n"
        << "  --validate        Check calibration and undistortion invariants\n"
        << "  -h, --help        Show this help text\n";
}

// Convert process arguments into a validated Options value.
Options parse_options(const int argc, char** argv) {
    // Begin with safe, documented defaults.
    Options options;
    // argv[0] names the executable, so start parsing at index one.
    for (int index = 1; index < argc; ++index) {
        // Copy the token before comparing it with supported option names.
        const std::string argument = argv[index];
        // All three path options require exactly one following value.
        if ((argument == "--images" ||
             argument == "--sample" ||
             argument == "--output-dir") &&
            index + 1 >= argc) {
            throw std::invalid_argument(argument + " requires a value");
        }

        if (argument == "--images") {
            // Increment first so the wildcard is not parsed again as an option.
            options.images = argv[++index];
        } else if (argument == "--sample") {
            // The sample must have the calibration image resolution.
            options.sample = argv[++index];
        } else if (argument == "--output-dir") {
            // Nested output directories are created by the shared helper.
            options.output_dir = argv[++index];
        } else if (argument == "--no-display") {
            // Headless mode prevents every HighGUI call.
            options.display = false;
        } else if (argument == "--validate") {
            // Validate the camera model and agreement between both methods.
            options.validate = true;
        } else if (argument == "-h" || argument == "--help") {
            // Defer the successful exit until main can print usage.
            options.show_help = true;
        } else {
            // Reject option typos instead of silently running with defaults.
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    // Return one complete configuration to main.
    return options;
}

// Run the calibration-plus-undistortion lesson.
int main(const int argc, char** argv) {
    try {
        // Resolve and validate all requested behavior before touching files.
        Options options = parse_options(argc, argv);
        // Help is successful and deliberately side-effect free.
        if (options.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        // Estimate intrinsics, distortion, and checkerboard poses.
        const tutorial::CalibrationResult calibration =
            tutorial::calibrate_images(options.images, options.display);
        // Report the same camera values as the calibration-only lesson.
        tutorial::print_calibration(calibration);

        // Sorting makes the first calibration image a deterministic default.
        if (options.sample.empty()) {
            options.sample = calibration.image_paths.front();
        }
        // Generate direct and precomputed-map undistortion results.
        const tutorial::UndistortionResult undistortion =
            tutorial::undistort_sample(
                calibration,
                options.sample,
                options.output_dir,
                options.display
            );
        // Print enough precision to expose the OpenCV 5 interpolation difference.
        std::cout
            << "Undistortion method mean absolute difference: "
            << undistortion.mean_absolute_difference << '\n';

        // Custom exploratory runs need not meet the bundled validation contract.
        if (options.validate) {
            // Reject malformed or poor-quality camera parameters first.
            tutorial::validate_calibration(calibration);
            // Check matrices, method agreement, and both decoded output files.
            tutorial::validate_undistortion(
                undistortion,
                kMaximumMethodDifference
            );
            // CTest requires this marker only after every check succeeds.
            std::cout << "Validation passed\n";
        }
        // Zero signals that calibration, output, and requested checks succeeded.
        return 0;
    } catch (const std::exception& error) {
        // Convert option, image, OpenCV, filesystem, and validation failures.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
