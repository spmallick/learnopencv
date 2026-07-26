// Original example by Sunita Nayak at BigVision LLC, based on OpenCV.
// Modernized for current OpenCV APIs and reproducible headless execution.

#include <opencv2/core/utility.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const char* kKeys =
    "{help h usage ? |      | Show this message }"
    "{image i        |      | Input image; defaults to test.jpg }"
    "{video v        |      | Input video }"
    "{camera c       |-1    | Camera device index }"
    "{overlay        |new_scenery.jpg| Image placed in the marker region }"
    "{output o       |      | Output image or AVI path }"
    "{augmented-only |false | Write only the augmented view }"
    "{strict         |false | Fail if no frame can be augmented }"
    "{display        |false | Show a GUI window }";

struct AugmentResult {
    cv::Mat image;
    bool augmented = false;
    std::vector<int> detected_ids;
};

std::optional<std::size_t> markerIndex(
    const std::vector<int>& ids,
    int marker_id
) {
    const auto iterator = std::find(ids.begin(), ids.end(), marker_id);
    if (iterator == ids.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(ids.begin(), iterator));
}

std::optional<std::vector<cv::Point2f>> destinationPoints(
    const std::vector<std::vector<cv::Point2f>>& marker_corners,
    const std::vector<int>& marker_ids,
    float margin_fraction = 0.02F
) {
    const auto top_left_index = markerIndex(marker_ids, 25);
    const auto top_right_index = markerIndex(marker_ids, 33);
    const auto bottom_right_index = markerIndex(marker_ids, 30);
    const auto bottom_left_index = markerIndex(marker_ids, 23);
    if (!top_left_index || !top_right_index ||
        !bottom_right_index || !bottom_left_index) {
        return std::nullopt;
    }

    const cv::Point2f top_left = marker_corners.at(*top_left_index).at(1);
    const cv::Point2f top_right = marker_corners.at(*top_right_index).at(2);
    const cv::Point2f bottom_right =
        marker_corners.at(*bottom_right_index).at(0);
    const cv::Point2f bottom_left = marker_corners.at(*bottom_left_index).at(0);
    const float top_edge_length = cv::norm(top_left - top_right);
    if (top_edge_length <= 1.0F) {
        throw std::runtime_error(
            "The top marker corners are too close to define a region."
        );
    }
    const float margin = margin_fraction * top_edge_length;
    return std::vector<cv::Point2f>{
        {top_left.x - margin, top_left.y - margin},
        {top_right.x + margin, top_right.y - margin},
        {bottom_right.x + margin, bottom_right.y + margin},
        {bottom_left.x - margin, bottom_left.y + margin},
    };
}

AugmentResult augmentFrame(
    const cv::Mat& frame,
    const cv::Mat& overlay,
    const cv::aruco::ArucoDetector& detector
) {
    if (frame.empty()) {
        throw std::invalid_argument("The input frame is empty.");
    }
    if (overlay.empty()) {
        throw std::invalid_argument("The overlay image is empty.");
    }

    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<std::vector<cv::Point2f>> rejected_candidates;
    std::vector<int> marker_ids;
    detector.detectMarkers(
        frame, marker_corners, marker_ids, rejected_candidates
    );

    AugmentResult result{frame.clone(), false, marker_ids};
    std::sort(result.detected_ids.begin(), result.detected_ids.end());
    const auto points_destination =
        destinationPoints(marker_corners, marker_ids);
    if (!points_destination) {
        return result;
    }

    const std::vector<cv::Point2f> points_source{
        {0.0F, 0.0F},
        {static_cast<float>(overlay.cols - 1), 0.0F},
        {
            static_cast<float>(overlay.cols - 1),
            static_cast<float>(overlay.rows - 1)
        },
        {0.0F, static_cast<float>(overlay.rows - 1)},
    };
    const cv::Mat transform =
        cv::getPerspectiveTransform(points_source, *points_destination);
    cv::Mat warped;
    cv::warpPerspective(
        overlay, warped, transform, frame.size(), cv::INTER_CUBIC
    );

    std::vector<cv::Point> polygon;
    polygon.reserve(points_destination->size());
    for (const cv::Point2f& point : *points_destination) {
        polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::fillConvexPoly(mask, polygon, cv::Scalar(255), cv::LINE_AA);
    const cv::Mat erosion_element =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(mask, mask, erosion_element, cv::Point(-1, -1), 3);

    warped.copyTo(result.image, mask);
    result.augmented = true;
    return result;
}

cv::Mat composeOutput(
    const cv::Mat& original,
    const cv::Mat& augmented,
    bool augmented_only
) {
    if (augmented_only) {
        return augmented;
    }
    cv::Mat side_by_side;
    cv::hconcat(original, augmented, side_by_side);
    return side_by_side;
}

fs::path imageOutputPath(const fs::path& input_path) {
    const std::string extension =
        input_path.has_extension() ? input_path.extension().string() : ".jpg";
    return input_path.parent_path() /
           fs::path(input_path.stem().string() + "_ar_out_cpp" + extension);
}

fs::path videoOutputPath(const fs::path& input_path) {
    if (input_path.empty()) {
        return "ar_out_cpp.avi";
    }
    return input_path.parent_path() /
           fs::path(input_path.stem().string() + "_ar_out_cpp.avi");
}

void createParentDirectory(const fs::path& output_path) {
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }
}

void printDetectedIds(const std::vector<int>& ids) {
    if (ids.empty()) {
        std::cout << "none";
        return;
    }
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index > 0) {
            std::cout << ',';
        }
        std::cout << ids[index];
    }
}

}  // namespace

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, kKeys);
    parser.about("Augmented reality with four ArUco markers");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 2;
    }

    std::string image_path = parser.get<std::string>("image");
    const std::string video_path = parser.get<std::string>("video");
    const int camera = parser.get<int>("camera");
    const std::string overlay_path = parser.get<std::string>("overlay");
    const std::string requested_output = parser.get<std::string>("output");
    const bool augmented_only = parser.get<bool>("augmented-only");
    const bool strict = parser.get<bool>("strict");
    const bool display = parser.get<bool>("display");

    const int input_count =
        static_cast<int>(!image_path.empty()) +
        static_cast<int>(!video_path.empty()) +
        static_cast<int>(camera >= 0);
    if (input_count > 1) {
        std::cerr << "Error: choose only one of --image, --video, or --camera.\n";
        return 2;
    }
    if (input_count == 0) {
        image_path = "test.jpg";
    }

    const cv::Mat overlay = cv::imread(overlay_path, cv::IMREAD_COLOR);
    if (overlay.empty()) {
        std::cerr << "Error: overlay image not found or unreadable: "
                  << overlay_path << '\n';
        return 2;
    }

    const cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    const cv::aruco::DetectorParameters parameters;
    const cv::aruco::ArucoDetector detector(dictionary, parameters);

    try {
        if (!image_path.empty()) {
            const cv::Mat frame =
                cv::imread(image_path, cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "Error: input image not found or unreadable: "
                          << image_path << '\n';
                return 2;
            }
            const AugmentResult result =
                augmentFrame(frame, overlay, detector);
            if (strict && !result.augmented) {
                std::cerr << "Error: required marker IDs 25, 33, 30, and 23 "
                             "were not all detected.\n";
                return 3;
            }
            const cv::Mat output =
                composeOutput(frame, result.image, augmented_only);
            const fs::path output_path = requested_output.empty()
                ? imageOutputPath(image_path)
                : fs::path(requested_output);
            createParentDirectory(output_path);
            if (!cv::imwrite(output_path.string(), output)) {
                std::cerr << "Error: OpenCV could not write: "
                          << output_path << '\n';
                return 3;
            }
            std::cout << "Wrote " << output_path << ": "
                      << (result.augmented ? "augmented" : "unchanged")
                      << "; detected IDs ";
            printDetectedIds(result.detected_ids);
            std::cout << '\n';

            if (display) {
                cv::imshow("AR using ArUco markers", output);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
            return 0;
        }

        cv::VideoCapture capture;
        if (!video_path.empty()) {
            capture.open(video_path);
        } else {
            capture.open(camera);
        }
        if (!capture.isOpened()) {
            std::cerr << "Error: could not open "
                      << (!video_path.empty() ? video_path : "camera") << ".\n";
            return 2;
        }

        const fs::path output_path = requested_output.empty()
            ? videoOutputPath(video_path)
            : fs::path(requested_output);
        createParentDirectory(output_path);
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) {
            fps = 28.0;
        }

        cv::VideoWriter writer;
        int frame_count = 0;
        int augmented_count = 0;
        while (true) {
            cv::Mat frame;
            if (!capture.read(frame)) {
                break;
            }
            const AugmentResult result =
                augmentFrame(frame, overlay, detector);
            const cv::Mat output =
                composeOutput(frame, result.image, augmented_only);
            if (!writer.isOpened()) {
                writer.open(
                    output_path.string(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps,
                    output.size()
                );
                if (!writer.isOpened()) {
                    std::cerr << "Error: could not create video: "
                              << output_path << '\n';
                    return 3;
                }
            }
            writer.write(output);
            ++frame_count;
            augmented_count += static_cast<int>(result.augmented);

            if (display) {
                cv::imshow("AR using ArUco markers", output);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    break;
                }
            }
        }
        capture.release();
        writer.release();
        if (display) {
            cv::destroyAllWindows();
        }

        if (frame_count == 0) {
            std::cerr << "Error: the input produced no readable frames.\n";
            return 3;
        }
        if (strict && augmented_count == 0) {
            std::cerr << "Error: no frame contained all required markers.\n";
            return 3;
        }
        std::cout << "Wrote " << output_path << ": augmented "
                  << augmented_count << '/' << frame_count << " frames\n";
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 3;
    }
    return 0;
}
