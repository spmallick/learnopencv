// This code is written by Sunita Nayak at BigVision LLC.
// It is subject to the license terms in the LICENSE file in this folder.

#include "colorization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const colorization::CommonOptions options = colorization::parseOptions(
            argc,
            argv,
            "greyscaleVideo.mp4",
            "colorized-video.avi",
            true);

        cv::VideoCapture capture(options.input.string());
        if (!capture.isOpened()) {
            throw std::runtime_error(
                "Could not open input video: " + options.input.string());
        }
        const int width =
            static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height =
            static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps < 1.0 || fps > 240.0) {
            fps = 30.0;
        }

        const std::filesystem::path parent = options.output.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        cv::VideoWriter writer(
            options.output.string(),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            cv::Size(width, height));
        if (!writer.isOpened()) {
            throw std::runtime_error(
                "Could not open output video: " + options.output.string());
        }

        cv::dnn::Net network = colorization::loadNetwork(options.model);
        int processed = 0;
        double inferenceSeconds = 0.0;
        cv::Mat frame;
        while (capture.read(frame)) {
            const auto start = std::chrono::steady_clock::now();
            auto [output, chromaScore] =
                colorization::colorizeFrame(frame, network);
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - start;
            inferenceSeconds += elapsed.count();
            if (options.validate) {
                colorization::validateOutput(frame, output, chromaScore);
            }

            writer.write(output);
            ++processed;
            if (!options.noDisplay) {
                cv::imshow("Colorized video", output);
                if ((cv::waitKey(1) & 0xFF) == 27) {
                    break;
                }
            }
            if (options.maxFrames > 0 && processed >= options.maxFrames) {
                break;
            }
        }

        capture.release();
        writer.release();
        if (!options.noDisplay) {
            cv::destroyAllWindows();
        }
        if (options.validate && processed == 0) {
            throw std::runtime_error("No frames were processed.");
        }

        const double average =
            processed > 0 ? inferenceSeconds / processed : 0.0;
        std::cout << "Saved " << processed << " frames to " << options.output
                  << '\n'
                  << "Average inference time: " << std::fixed
                  << std::setprecision(3) << average
                  << " seconds per frame\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
