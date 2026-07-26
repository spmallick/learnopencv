#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

const char* kKeys =
    "{help h usage ? |      | Show this message }"
    "{input i        |image.png| Input image }"
    "{model m        |models/ESPCN_x4.pb| TensorFlow .pb model }"
    "{algorithm a    |espcn| Model architecture: edsr, espcn, fsrcnn, or lapsrn }"
    "{scale s        |4     | Upscaling factor encoded by the model }"
    "{output o       |output.png| Output image }";

bool isSupportedConfiguration(const std::string& algorithm, int scale) {
    if (algorithm == "lapsrn") {
        return scale == 2 || scale == 4 || scale == 8;
    }
    if (algorithm == "edsr" || algorithm == "espcn" ||
        algorithm == "fsrcnn") {
        return scale == 2 || scale == 3 || scale == 4;
    }
    return false;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, kKeys);
    parser.about("OpenCV DNN super-resolution example");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    const std::string input_path = parser.get<std::string>("input");
    const std::string model_path = parser.get<std::string>("model");
    const std::string algorithm = parser.get<std::string>("algorithm");
    const int scale = parser.get<int>("scale");
    const fs::path output_path = parser.get<std::string>("output");

    if (!isSupportedConfiguration(algorithm, scale)) {
        std::cerr << "Error: unsupported --algorithm/--scale combination. "
                     "Use edsr, espcn, or fsrcnn with x2/x3/x4, or lapsrn "
                     "with x2/x4/x8.\n";
        return 2;
    }
    if (!fs::is_regular_file(model_path)) {
        std::cerr << "Error: model not found: " << model_path
                  << ". Run 'python download_model.py' or pass --model.\n";
        return 2;
    }

    const cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: input image not found or unreadable: "
                  << input_path << '\n';
        return 2;
    }

    try {
        cv::dnn_superres::DnnSuperResImpl super_resolver;
        super_resolver.readModel(model_path);
        super_resolver.setModel(algorithm, scale);

        cv::Mat result;
        super_resolver.upsample(image, result);
        const cv::Size expected(image.cols * scale, image.rows * scale);
        if (result.size() != expected) {
            std::cerr << "Error: unexpected output dimensions: "
                      << result.cols << 'x' << result.rows << "; expected "
                      << expected.width << 'x' << expected.height << '\n';
            return 3;
        }

        if (output_path.has_parent_path()) {
            fs::create_directories(output_path.parent_path());
        }
        if (!cv::imwrite(output_path.string(), result)) {
            std::cerr << "Error: OpenCV could not write: " << output_path << '\n';
            return 3;
        }
        std::cout << "Upscaled " << image.cols << 'x' << image.rows << " -> "
                  << result.cols << 'x' << result.rows << " and wrote "
                  << output_path << '\n';
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 3;
    } catch (const fs::filesystem_error& error) {
        std::cerr << "Filesystem error: " << error.what() << '\n';
        return 3;
    }

    return 0;
}
