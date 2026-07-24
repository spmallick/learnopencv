// Shared OpenCV 4.14/5.0 calibration logic lives in one teaching-oriented header.
#include "calibration_utils.hpp"

// Filesystem joins the compiled asset directory and wildcard portably.
#include <filesystem>
// The CLI prints help, calibration values, validation markers, and errors.
#include <iostream>
// Invalid options become readable command-line failures.
#include <stdexcept>
// Strings store option names and the selected image pattern.
#include <string>

// Keep path expressions concise in the small option structure.
namespace fs = std::filesystem;

// Hold command-line choices separately from camera-calibration results.
struct Options {
    // CMake embeds the source image directory so any working directory succeeds.
    std::string images =
        (fs::path(TUTORIAL_IMAGE_DIR) / "*.jpg").string();
    // Preserve the original interactive corner preview by default.
    bool display = true;
    // Make regression checks opt-in for normal tutorial exploration.
    bool validate = false;
    // Help exits before loading any image.
    bool show_help = false;
};

// Print the exact interface accepted by this executable.
void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " [--images GLOB] [--no-display] [--validate]\n"
        << "\n"
        << "Options:\n"
        << "  --images GLOB  Calibration image pattern (default: bundled images)\n"
        << "  --no-display   Skip interactive checkerboard windows\n"
        << "  --validate     Check calibration invariants\n"
        << "  -h, --help     Show this help text\n";
}

// Convert process arguments into a validated Options value.
Options parse_options(const int argc, char** argv) {
    // Begin with bundled images, display enabled, and validation disabled.
    Options options;
    // argv[0] is the executable name, so option parsing starts at index one.
    for (int index = 1; index < argc; ++index) {
        // Copy the current token so comparisons remain clear and safe.
        const std::string argument = argv[index];
        // The image option must consume exactly one following glob string.
        if (argument == "--images") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--images requires a glob pattern"
                );
            }
            // Increment before reading so the value is not parsed as an option.
            options.images = argv[++index];
        } else if (argument == "--no-display") {
            // Headless mode prevents all HighGUI calls in the shared helper.
            options.display = false;
        } else if (argument == "--validate") {
            // Validation checks the result after it has been printed.
            options.validate = true;
        } else if (argument == "-h" || argument == "--help") {
            // Record help instead of exiting inside the parser.
            options.show_help = true;
        } else {
            // Reject typos instead of silently ignoring an intended option.
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    // Return one complete configuration to main.
    return options;
}

// Run the public tutorial entry point.
int main(const int argc, char** argv) {
    try {
        // Validate the command line before accessing files or OpenCV.
        const Options options = parse_options(argc, argv);
        // Help is a successful, side-effect-free request.
        if (options.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        // Detect corners and estimate one camera model from the selected images.
        const tutorial::CalibrationResult calibration =
            tutorial::calibrate_images(options.images, options.display);
        // Print coverage, errors, intrinsics, and distortion coefficients.
        tutorial::print_calibration(calibration);
        // Keep regression checks optional for custom experimental datasets.
        if (options.validate) {
            tutorial::validate_calibration(calibration);
            // CTest requires this marker only after every check succeeds.
            std::cout << "Validation passed\n";
        }
        // Zero signals that all requested work completed successfully.
        return 0;
    } catch (const std::exception& error) {
        // Convert OpenCV, filesystem, option, and validation failures uniformly.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
