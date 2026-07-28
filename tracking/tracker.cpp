#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#if __has_include(<opencv2/tracking.hpp>)
#include <opencv2/tracking.hpp>
#define LEARNOPENCV_HAS_CONTRIB_TRACKING 1
#else
#define LEARNOPENCV_HAS_CONTRIB_TRACKING 0
#endif

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::toupper(character));
                   });
    return value;
}

cv::Ptr<cv::Tracker> createTracker(const std::string& requestedName) {
    const std::string name = upper(requestedName);
#if LEARNOPENCV_HAS_CONTRIB_TRACKING
    if (name == "CSRT") {
        return cv::TrackerCSRT::create();
    }
    if (name == "KCF") {
        return cv::TrackerKCF::create();
    }
#endif
    if (name == "MIL") {
        return cv::TrackerMIL::create();
    }
    throw std::invalid_argument(
        "Tracker '" + requestedName +
        "' is unavailable in this OpenCV build. MIL works in OpenCV 4 and 5; "
        "CSRT and KCF require the OpenCV 4 contrib tracking module.");
}

cv::Rect parseBoundingBox(const std::string& text) {
    std::stringstream stream(text);
    cv::Rect box;
    char separator1 = '\0';
    char separator2 = '\0';
    char separator3 = '\0';
    if (!(stream >> box.x >> separator1 >> box.y >> separator2
          >> box.width >> separator3 >> box.height) ||
        separator1 != ',' || separator2 != ',' || separator3 != ',' ||
        box.width <= 0 || box.height <= 0) {
        throw std::invalid_argument(
            "Bounding box must be x,y,width,height with positive dimensions.");
    }
    stream >> std::ws;
    if (!stream.eof()) {
        throw std::invalid_argument(
            "Bounding box must contain exactly four comma-separated integers.");
    }
    return box;
}

void validateBoundingBox(const cv::Rect& box, const cv::Size& frameSize) {
    if (box.x < 0 || box.y < 0 || box.width <= 0 || box.height <= 0 ||
        box.width > frameSize.width - box.x ||
        box.height > frameSize.height - box.y) {
        throw std::invalid_argument("Bounding box lies outside the first frame.");
    }
}

cv::VideoWriter openWriter(
    const std::string& path, double fps, const cv::Size& frameSize) {
    const bool isMp4 =
        path.size() >= 4 && path.substr(path.size() - 4) == ".mp4";
    const int codec = isMp4
        ? cv::VideoWriter::fourcc('m', 'p', '4', 'v')
        : cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer(path, codec, fps, frameSize);
    if (!writer.isOpened()) {
        throw std::runtime_error("Could not open output video: " + path);
    }
    return writer;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string keys =
        "{help h usage ?||Show this help message}"
        "{input|videos/chaplin.mp4|Input video path}"
        "{tracker|MIL|MIL works in OpenCV 4/5; CSRT/KCF require OpenCV 4 contrib}"
        "{bbox|287,23,86,320|Initial x,y,width,height}"
        "{select-roi||Select the initial box interactively}"
        "{output||Optional annotated output video}"
        "{snapshot||Optional final annotated frame}"
        "{max-frames|-1|Maximum update frames; -1 processes all}"
        "{display||Show an interactive window}";
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
        const std::string trackerName = upper(parser.get<std::string>("tracker"));
        const std::string outputPath = parser.get<std::string>("output");
        const std::string snapshotPath = parser.get<std::string>("snapshot");
        const int maxFrames = parser.get<int>("max-frames");
        const bool display = parser.has("display");
        const bool selectRoi = parser.has("select-roi");
        if (maxFrames == 0 || maxFrames < -1) {
            throw std::invalid_argument("max-frames must be positive or -1.");
        }

        cv::VideoCapture capture(inputPath);
        if (!capture.isOpened()) {
            throw std::runtime_error("Could not open input video: " + inputPath);
        }

        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            throw std::runtime_error("Could not read the first input frame.");
        }

        cv::Rect box = selectRoi
            ? cv::selectROI("Select object", frame, false, false)
            : parseBoundingBox(parser.get<std::string>("bbox"));
        if (selectRoi) {
            cv::destroyWindow("Select object");
        }
        validateBoundingBox(box, frame.size());

        cv::Ptr<cv::Tracker> tracker = createTracker(trackerName);
        tracker->init(frame, box);

        double sourceFps = capture.get(cv::CAP_PROP_FPS);
        if (!(sourceFps > 0.0)) {
            sourceFps = 30.0;
        }
        cv::VideoWriter writer;
        if (!outputPath.empty()) {
            writer = openWriter(outputPath, sourceFps, frame.size());
        }

        int framesProcessed = 0;
        int successfulUpdates = 0;
        double elapsedTotal = 0.0;
        cv::Mat lastAnnotated;
        while ((maxFrames < 0 || framesProcessed < maxFrames) &&
               capture.read(frame) && !frame.empty()) {
            const int64 start = cv::getTickCount();
            const bool found = tracker->update(frame, box);
            const double elapsed =
                (cv::getTickCount() - start) / cv::getTickFrequency();
            elapsedTotal += elapsed;
            ++framesProcessed;

            if (found) {
                cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
                ++successfulUpdates;
            } else {
                cv::putText(
                    frame, "Tracking failure detected", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }

            const double instantaneousFps =
                elapsed > 0.0 ? 1.0 / elapsed : 0.0;
            std::ostringstream label;
            label << trackerName << " | " << std::fixed
                  << std::setprecision(1) << instantaneousFps << " FPS";
            cv::putText(
                frame, label.str(), cv::Point(20, frame.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 170, 50), 2);
            lastAnnotated = frame.clone();

            if (writer.isOpened()) {
                writer.write(frame);
            }
            if (display) {
                cv::imshow("Object tracking", frame);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q') {
                    break;
                }
            }
        }

        if (framesProcessed == 0) {
            throw std::runtime_error(
                "The input contains no frames after initialization.");
        }
        if (!snapshotPath.empty() &&
            !cv::imwrite(snapshotPath, lastAnnotated)) {
            throw std::runtime_error(
                "Could not write snapshot: " + snapshotPath);
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "{\"tracker\":\"" << trackerName
                  << "\",\"frames_processed\":" << framesProcessed
                  << ",\"successful_updates\":" << successfulUpdates
                  << ",\"success_rate\":"
                  << static_cast<double>(successfulUpdates) / framesProcessed
                  << ",\"last_bbox\":["
                  << box.x << ',' << box.y << ',' << box.width << ',' << box.height
                  << "],\"frame_size\":[" << frame.cols << ',' << frame.rows
                  << "],\"source_fps\":" << sourceFps
                  << ",\"mean_tracking_ms\":"
                  << 1000.0 * elapsedTotal / framesProcessed << "}\n";
        return 0;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
    }
    return 2;
}
