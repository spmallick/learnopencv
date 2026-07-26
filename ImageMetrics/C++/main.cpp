#include <opencv2/imgcodecs.hpp>
#include <opencv2/quality/qualitybrisque.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef BRISQUE_DEFAULT_MODEL
#define BRISQUE_DEFAULT_MODEL "../models/brisque_model_live.yml"
#endif

#ifndef BRISQUE_DEFAULT_RANGE
#define BRISQUE_DEFAULT_RANGE "../models/brisque_range_live.yml"
#endif

namespace {

struct Arguments {
  std::filesystem::path image;
  std::filesystem::path model = BRISQUE_DEFAULT_MODEL;
  std::filesystem::path range = BRISQUE_DEFAULT_RANGE;
};

void printUsage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " IMAGE [--model MODEL_YML] [--range RANGE_YML]\n";
}

Arguments parseArguments(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument("Input image argument not given.");
  }

  Arguments arguments;
  arguments.image = argv[1];

  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    if ((option == "--model" || option == "--range") && index + 1 < argc) {
      const std::filesystem::path value = argv[++index];
      if (option == "--model") {
        arguments.model = value;
      } else {
        arguments.range = value;
      }
      continue;
    }
    throw std::invalid_argument("Unknown or incomplete option: " + option);
  }

  return arguments;
}

void requireFile(const std::filesystem::path& path, const std::string& label) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("Could not read " + label + ": " + path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
    requireFile(arguments.image, "image");
    requireFile(arguments.model, "BRISQUE model");
    requireFile(arguments.range, "BRISQUE range");

    const cv::Mat image =
        cv::imread(arguments.image.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error(
          "OpenCV could not decode image: " + arguments.image.string());
    }

    const cv::Scalar score = cv::quality::QualityBRISQUE::compute(
        image, arguments.model.string(), arguments.range.string());
    std::cout << "BRISQUE score: " << std::fixed << std::setprecision(4)
              << score[0] << '\n';
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n';
    printUsage(argv[0]);
    return 2;
  } catch (const cv::Exception& error) {
    std::cerr << "OpenCV error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
