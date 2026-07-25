#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Options {
    fs::path template_path = "images/image1.jpg";
    fs::path moving_path = "images/image2.jpg";
    fs::path output = "output/image2-aligned.jpg";
    std::string motion = "euclidean";
    int iterations = 5000;
    double epsilon = 1e-7;
    bool display = true;
    bool validate = false;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--template" && i + 1 < argc) {
            options.template_path = argv[++i];
        } else if (argument == "--moving" && i + 1 < argc) {
            options.moving_path = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (argument == "--motion" && i + 1 < argc) {
            options.motion = argv[++i];
        } else if (argument == "--iterations" && i + 1 < argc) {
            options.iterations = std::stoi(argv[++i]);
        } else if (argument == "--epsilon" && i + 1 < argc) {
            options.epsilon = std::stod(argv[++i]);
        } else if (argument == "--no-display") {
            options.display = false;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--help") {
            std::cout
                << "Usage: image_alignment_simple [--template FILE] "
                   "[--moving FILE] [--motion translation|euclidean|affine|homography] "
                   "[--output FILE] [--no-display] [--validate]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

double meanAbsoluteError(const cv::Mat& first, const cv::Mat& second) {
    cv::Mat difference;
    cv::absdiff(first, second, difference);
    return cv::mean(difference)[0] + cv::mean(difference)[1] +
           cv::mean(difference)[2];
}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const std::map<std::string, int> motion_models = {
            {"translation", cv::MOTION_TRANSLATION},
            {"euclidean", cv::MOTION_EUCLIDEAN},
            {"affine", cv::MOTION_AFFINE},
            {"homography", cv::MOTION_HOMOGRAPHY},
        };
        const auto motion_iterator = motion_models.find(options.motion);
        if (motion_iterator == motion_models.end()) {
            throw std::invalid_argument("Unknown motion model: " + options.motion);
        }
        const int motion_model = motion_iterator->second;

        const cv::Mat template_image = cv::imread(options.template_path.string());
        cv::Mat moving_image = cv::imread(options.moving_path.string());
        if (template_image.empty() || moving_image.empty()) {
            throw std::runtime_error("Could not read one or both input images");
        }
        if (template_image.size() != moving_image.size()) {
            cv::resize(moving_image, moving_image, template_image.size(), 0.0, 0.0,
                       cv::INTER_AREA);
        }
        cv::Mat template_gray;
        cv::Mat moving_gray;
        cv::cvtColor(template_image, template_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(moving_image, moving_gray, cv::COLOR_BGR2GRAY);
        template_gray.convertTo(template_gray, CV_32F);
        moving_gray.convertTo(moving_gray, CV_32F);

        cv::Mat warp = motion_model == cv::MOTION_HOMOGRAPHY
                           ? cv::Mat::eye(3, 3, CV_32F)
                           : cv::Mat::eye(2, 3, CV_32F);
        const cv::TermCriteria criteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS, options.iterations,
            options.epsilon);
        const double correlation =
            cv::findTransformECC(template_gray, moving_gray, warp, motion_model,
                                 criteria);

        cv::Mat aligned;
        const int flags = cv::INTER_LINEAR | cv::WARP_INVERSE_MAP;
        if (motion_model == cv::MOTION_HOMOGRAPHY) {
            cv::warpPerspective(moving_image, aligned, warp, template_image.size(),
                                flags);
        } else {
            cv::warpAffine(moving_image, aligned, warp, template_image.size(),
                           flags);
        }
        const double before = meanAbsoluteError(template_image, moving_image);
        const double after = meanAbsoluteError(template_image, aligned);
        if (!options.output.parent_path().empty()) {
            fs::create_directories(options.output.parent_path());
        }
        if (!cv::imwrite(options.output.string(), aligned)) {
            throw std::runtime_error("Could not write aligned image");
        }

        std::cout << "OpenCV: " << CV_VERSION << "\n"
                  << "Motion model: " << options.motion << "\n"
                  << "ECC correlation: " << correlation << "\n"
                  << "Warp matrix:\n" << warp << "\n"
                  << "Mean absolute error: " << before << " -> " << after << "\n"
                  << "Saved: " << options.output << "\n";

        if (options.validate) {
            if (!cv::checkRange(warp) || !std::isfinite(correlation) ||
                correlation <= 0.0 || after >= before) {
                throw std::runtime_error("ECC pair-alignment validation failed");
            }
            std::cout << "Validation: PASS\n";
        }
        if (options.display) {
            cv::imshow("Template", template_image);
            cv::imshow("Moving", moving_image);
            cv::imshow("Aligned", aligned);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
