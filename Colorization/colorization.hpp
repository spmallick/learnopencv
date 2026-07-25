#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace colorization {

inline cv::dnn::Net loadNetwork(const std::filesystem::path& modelPath) {
    if (!std::filesystem::is_regular_file(modelPath)) {
        throw std::runtime_error(
            "Model not found: " + modelPath.string() +
            ". Run ./getModels.sh before the demo.");
    }

    cv::dnn::Net network = cv::dnn::readNetFromONNX(modelPath.string());
    network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return network;
}

inline std::pair<cv::Mat, double> colorizeFrame(
    const cv::Mat& frameBgr,
    cv::dnn::Net& network) {
    if (frameBgr.empty() || frameBgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "Expected a non-empty, 8-bit, three-channel BGR frame.");
    }

    cv::Mat image;
    frameBgr.convertTo(image, CV_32F, 1.0 / 255.0);

    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    cv::Mat lightness;
    cv::extractChannel(lab, lightness, 0);

    cv::Mat resizedLightness;
    cv::resize(
        lightness,
        resizedLightness,
        cv::Size(256, 256),
        0.0,
        0.0,
        cv::INTER_CUBIC);
    network.setInput(cv::dnn::blobFromImage(resizedLightness));
    cv::Mat prediction = network.forward();
    if (prediction.dims != 4 || prediction.size[0] != 1 ||
        prediction.size[1] != 2) {
        throw std::runtime_error("The ONNX model returned an unexpected shape.");
    }

    const cv::Size networkOutputSize(prediction.size[3], prediction.size[2]);
    const cv::Mat predictedA(
        networkOutputSize, CV_32F, prediction.ptr<float>(0, 0));
    const cv::Mat predictedB(
        networkOutputSize, CV_32F, prediction.ptr<float>(0, 1));
    cv::Mat a;
    cv::Mat b;
    cv::resize(predictedA, a, frameBgr.size(), 0.0, 0.0, cv::INTER_CUBIC);
    cv::resize(predictedB, b, frameBgr.size(), 0.0, 0.0, cv::INTER_CUBIC);

    std::vector<cv::Mat> channels{lightness, a, b};
    cv::merge(channels, lab);
    cv::Mat output;
    cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);
    output *= 255.0F;
    output.convertTo(output, CV_8UC3);

    cv::Mat magnitude;
    cv::magnitude(a, b, magnitude);
    const double chromaScore = cv::mean(magnitude)[0];
    return {output, chromaScore};
}

inline void validateOutput(
    const cv::Mat& input,
    const cv::Mat& output,
    const double chromaScore) {
    if (output.size() != input.size() || output.type() != CV_8UC3) {
        throw std::runtime_error(
            "The generated output does not match the input dimensions and type.");
    }
    if (!std::isfinite(chromaScore) || chromaScore <= 1.0) {
        throw std::runtime_error("The generated output contains too little chroma.");
    }
}

struct CommonOptions {
    std::filesystem::path input;
    std::filesystem::path model = "models/colorization_eccv16.onnx";
    std::filesystem::path output;
    bool noDisplay = false;
    bool validate = false;
    int maxFrames = 0;
};

inline CommonOptions parseOptions(
    const int argc,
    char** argv,
    const std::filesystem::path& defaultInput,
    const std::filesystem::path& defaultOutput,
    const bool allowMaxFrames) {
    CommonOptions options;
    options.input = defaultInput;
    options.output = defaultOutput;

    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        const auto requireValue = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value after " + argument);
            }
            return argv[++index];
        };

        if (argument == "--input") {
            options.input = requireValue();
        } else if (argument == "--model") {
            options.model = requireValue();
        } else if (argument == "--output") {
            options.output = requireValue();
        } else if (argument == "--no-display") {
            options.noDisplay = true;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--max-frames" && allowMaxFrames) {
            options.maxFrames = std::stoi(requireValue());
            if (options.maxFrames < 0) {
                throw std::runtime_error("--max-frames cannot be negative.");
            }
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
    return options;
}

}  // namespace colorization
