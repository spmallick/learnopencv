#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

const char* kKeys =
    "{help h usage ? |      | Show this message }"
    "{id             |33    | Marker ID from 0 to 249 }"
    "{size           |200   | Square image size in pixels }"
    "{border-bits    |1     | Marker border width in bits }"
    "{output o       |marker33.png| Output PNG path }";

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, kKeys);
    parser.about("Generate a DICT_6X6_250 ArUco marker");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    const int marker_id = parser.get<int>("id");
    const int size = parser.get<int>("size");
    const int border_bits = parser.get<int>("border-bits");
    const fs::path output_path = parser.get<std::string>("output");
    if (marker_id < 0 || marker_id >= 250) {
        std::cerr << "Error: --id must be between 0 and 249.\n";
        return 2;
    }
    if (size < 32) {
        std::cerr << "Error: --size must be at least 32 pixels.\n";
        return 2;
    }
    if (border_bits < 1) {
        std::cerr << "Error: --border-bits must be at least 1.\n";
        return 2;
    }

    try {
        const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::Mat marker_image;
        dictionary.generateImageMarker(
            marker_id, size, marker_image, border_bits
        );
        if (output_path.has_parent_path()) {
            fs::create_directories(output_path.parent_path());
        }
        if (!cv::imwrite(output_path.string(), marker_image)) {
            std::cerr << "Error: OpenCV could not write: "
                      << output_path << '\n';
            return 3;
        }
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 3;
    } catch (const fs::filesystem_error& error) {
        std::cerr << "Filesystem error: " << error.what() << '\n';
        return 3;
    }

    std::cout << "Wrote marker " << marker_id << " (" << size << 'x'
              << size << ") to " << output_path << '\n';
    return 0;
}
