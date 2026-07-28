#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    const std::string keys =
        "{help h usage ?||Show this help message}"
        "{input|chaplin.mp4|Input video path}"
        "{display||Show frames in a window}"
        "{max-frames|-1|Maximum frames; -1 reads all}";
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
        const bool display = parser.has("display");
        const int maxFrames = parser.get<int>("max-frames");
        if (maxFrames == 0 || maxFrames < -1) {
            throw std::invalid_argument("max-frames must be positive or -1.");
        }

        cv::VideoCapture capture(inputPath);
        if (!capture.isOpened()) {
            throw std::runtime_error("Could not open input video: " + inputPath);
        }

        cv::Mat frame;
        int framesRead = 0;
        std::uint64_t firstFrameChecksum = 0;
        while ((maxFrames < 0 || framesRead < maxFrames) &&
               capture.read(frame) && !frame.empty()) {
            ++framesRead;
            if (framesRead == 1) {
                firstFrameChecksum =
                    static_cast<std::uint64_t>(cv::sum(frame)[0]) +
                    static_cast<std::uint64_t>(cv::sum(frame)[1]) +
                    static_cast<std::uint64_t>(cv::sum(frame)[2]);
            }
            if (display) {
                cv::imshow("Video", frame);
                const int key = cv::waitKey(25) & 0xff;
                if (key == 27 || key == 'q') {
                    break;
                }
            }
        }
        if (framesRead == 0) {
            throw std::runtime_error("No decodable frames found.");
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "{\"frames_read\":" << framesRead
                  << ",\"frame_size\":["
                  << static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH))
                  << ','
                  << static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT))
                  << "],\"fps\":" << capture.get(cv::CAP_PROP_FPS)
                  << ",\"reported_frame_count\":"
                  << static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT))
                  << ",\"first_frame_checksum\":" << firstFrameChecksum
                  << "}\n";
        return 0;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
    }
    return 2;
}
