#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    const std::string keys =
        "{help h usage ?||Show this help message}"
        "{input|chaplin.mp4|Input video path}"
        "{output|output.avi|Output video path}"
        "{codec|MJPG|Four-character codec}"
        "{fps|-1|Output FPS; -1 preserves source FPS}"
        "{display||Show frames while writing}"
        "{max-frames|-1|Maximum frames; -1 writes all}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    try {
        const std::string inputPath = parser.get<std::string>("input");
        const std::string outputPath = parser.get<std::string>("output");
        const std::string codec = parser.get<std::string>("codec");
        const double requestedFps = parser.get<double>("fps");
        const int maxFrames = parser.get<int>("max-frames");
        const bool display = parser.has("display");
        if (codec.size() != 4) {
            throw std::invalid_argument(
                "codec must contain exactly four characters.");
        }
        if (!std::isfinite(requestedFps) ||
            (requestedFps != -1.0 && requestedFps <= 0.0)) {
            throw std::invalid_argument("fps must be positive or -1.");
        }
        if (maxFrames == 0 || maxFrames < -1) {
            throw std::invalid_argument("max-frames must be positive or -1.");
        }

        cv::VideoCapture capture(inputPath);
        if (!capture.isOpened()) {
            throw std::runtime_error("Could not open input video: " + inputPath);
        }
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            throw std::runtime_error("No decodable frames found.");
        }
        const cv::Size frameSize = frame.size();

        double sourceFps = capture.get(cv::CAP_PROP_FPS);
        if (!(sourceFps > 0.0)) {
            sourceFps = 30.0;
        }
        const double outputFps =
            requestedFps > 0.0 ? requestedFps : sourceFps;
        const int fourcc = cv::VideoWriter::fourcc(
            codec[0], codec[1], codec[2], codec[3]);
        cv::VideoWriter writer(
            outputPath, fourcc, outputFps, frameSize);
        if (!writer.isOpened()) {
            throw std::runtime_error(
                "Could not open output video: " + outputPath);
        }

        int framesWritten = 0;
        while (maxFrames < 0 || framesWritten < maxFrames) {
            writer.write(frame);
            ++framesWritten;
            if (display) {
                cv::imshow("Transcoding", frame);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q') {
                    break;
                }
            }
            if (!capture.read(frame) || frame.empty()) {
                break;
            }
        }
        writer.release();

        std::cout << std::fixed << std::setprecision(4)
                  << "{\"frames_written\":" << framesWritten
                  << ",\"frame_size\":[" << frameSize.width << ','
                  << frameSize.height
                  << "],\"source_fps\":" << sourceFps
                  << ",\"output_fps\":" << outputFps
                  << ",\"codec\":\"" << codec
                  << "\",\"output\":\"" << outputPath << "\"}\n";
        return 0;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
    }
    return 2;
}
