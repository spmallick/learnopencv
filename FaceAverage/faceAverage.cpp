#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    fs::path input_dir = "presidents";
    fs::path output = "output/average-face.jpg";
    int width = 600;
    int height = 600;
    bool display = true;
    bool validate = false;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--input-dir" && i + 1 < argc) {
            options.input_dir = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            options.output = argv[++i];
        } else if (argument == "--width" && i + 1 < argc) {
            options.width = std::stoi(argv[++i]);
        } else if (argument == "--height" && i + 1 < argc) {
            options.height = std::stoi(argv[++i]);
        } else if (argument == "--no-display") {
            options.display = false;
        } else if (argument == "--validate") {
            options.validate = true;
        } else if (argument == "--help") {
            std::cout
                << "Usage: face_average [--input-dir DIR] [--output FILE] "
                   "[--width N] [--height N] [--no-display] [--validate]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete argument: " + argument);
        }
    }
    if (options.width < 64 || options.height < 64) {
        throw std::invalid_argument("Output dimensions must be at least 64 pixels");
    }
    return options;
}

std::vector<cv::Point2f> readPoints(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Could not read landmarks: " + path.string());
    }
    std::vector<cv::Point2f> points;
    float x = 0.0F;
    float y = 0.0F;
    while (stream >> x >> y) {
        points.emplace_back(x, y);
    }
    if (points.size() != 68) {
        throw std::runtime_error("Expected 68 landmarks in " + path.string());
    }
    return points;
}

std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Point2f>>>
readDataset(const fs::path& directory) {
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Point2f>> point_sets;
    for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) {
            throw std::runtime_error("Could not read image: " + image_path.string());
        }
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);
        const fs::path points_path(image_path.string() + ".txt");
        images.push_back(image);
        point_sets.push_back(readPoints(points_path));
    }
    if (images.size() < 2) {
        throw std::runtime_error("Need at least two image/landmark pairs");
    }
    return {images, point_sets};
}

cv::Mat similarityTransform(const std::vector<cv::Point2f>& input,
                            const std::vector<cv::Point2f>& output) {
    const double radians = 60.0 * CV_PI / 180.0;
    const double sin60 = std::sin(radians);
    const double cos60 = std::cos(radians);
    std::vector<cv::Point2f> source = input;
    std::vector<cv::Point2f> target = output;
    source.emplace_back(
        static_cast<float>(cos60 * (source[0].x - source[1].x) -
                           sin60 * (source[0].y - source[1].y) + source[1].x),
        static_cast<float>(sin60 * (source[0].x - source[1].x) +
                           cos60 * (source[0].y - source[1].y) + source[1].y));
    target.emplace_back(
        static_cast<float>(cos60 * (target[0].x - target[1].x) -
                           sin60 * (target[0].y - target[1].y) + target[1].x),
        static_cast<float>(sin60 * (target[0].x - target[1].x) +
                           cos60 * (target[0].y - target[1].y) + target[1].y));
    cv::Mat transform = cv::estimateAffinePartial2D(source, target, cv::noArray(),
                                                     cv::LMEDS);
    if (transform.empty()) {
        throw std::runtime_error("Could not estimate a similarity transform");
    }
    return transform;
}

void calculateDelaunayTriangles(
    const cv::Rect& rect, const std::vector<cv::Point2f>& points,
    std::vector<cv::Vec3i>& triangles) {
    cv::Subdiv2D subdiv(rect);
    for (const auto& point : points) {
        subdiv.insert(point);
    }
    std::vector<cv::Vec6f> triangle_list;
    subdiv.getTriangleList(triangle_list);
    for (const auto& triangle : triangle_list) {
        const std::vector<cv::Point2f> vertices = {
            {triangle[0], triangle[1]},
            {triangle[2], triangle[3]},
            {triangle[4], triangle[5]},
        };
        if (!rect.contains(vertices[0]) || !rect.contains(vertices[1]) ||
            !rect.contains(vertices[2])) {
            continue;
        }
        cv::Vec3i indices(-1, -1, -1);
        for (int vertex = 0; vertex < 3; ++vertex) {
            float best_distance = 1.0F;
            for (std::size_t index = 0; index < points.size(); ++index) {
                const float distance = cv::norm(vertices[vertex] - points[index]);
                if (distance < best_distance) {
                    best_distance = distance;
                    indices[vertex] = static_cast<int>(index);
                }
            }
        }
        if (indices[0] >= 0 && indices[1] >= 0 && indices[2] >= 0 &&
            indices[0] != indices[1] && indices[1] != indices[2] &&
            indices[0] != indices[2]) {
            triangles.push_back(indices);
        }
    }
    if (triangles.empty()) {
        throw std::runtime_error("Delaunay triangulation produced no triangles");
    }
}

cv::Point2f constrainPoint(cv::Point2f point, const cv::Size& size) {
    point.x = std::clamp(point.x, 0.0F, static_cast<float>(size.width - 1));
    point.y = std::clamp(point.y, 0.0F, static_cast<float>(size.height - 1));
    return point;
}

void warpTriangle(const cv::Mat& source, cv::Mat& destination,
                  const std::vector<cv::Point2f>& source_triangle,
                  const std::vector<cv::Point2f>& target_triangle) {
    const cv::Rect source_rect = cv::boundingRect(source_triangle);
    const cv::Rect target_rect = cv::boundingRect(target_triangle);
    std::vector<cv::Point2f> source_local;
    std::vector<cv::Point2f> target_local;
    std::vector<cv::Point> target_local_int;
    for (int i = 0; i < 3; ++i) {
        source_local.emplace_back(source_triangle[i].x - source_rect.x,
                                  source_triangle[i].y - source_rect.y);
        target_local.emplace_back(target_triangle[i].x - target_rect.x,
                                  target_triangle[i].y - target_rect.y);
        target_local_int.emplace_back(
            cvRound(target_triangle[i].x - target_rect.x),
            cvRound(target_triangle[i].y - target_rect.y));
    }

    cv::Mat mask = cv::Mat::zeros(target_rect.height, target_rect.width, CV_32FC3);
    cv::fillConvexPoly(mask, target_local_int, cv::Scalar(1.0, 1.0, 1.0),
                       cv::LINE_AA);
    const cv::Mat source_patch = source(source_rect);
    cv::Mat warped;
    const cv::Mat transform = cv::getAffineTransform(source_local, target_local);
    cv::warpAffine(source_patch, warped, transform, target_rect.size(),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    cv::multiply(warped, mask, warped);
    cv::Mat target_view = destination(target_rect);
    cv::multiply(target_view, cv::Scalar(1.0, 1.0, 1.0) - mask, target_view);
    target_view += warped;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        auto [images, point_sets] = readDataset(options.input_dir);
        const cv::Size output_size(options.width, options.height);
        const std::vector<cv::Point2f> eye_target = {
            {0.3F * options.width, options.height / 3.0F},
            {0.7F * options.width, options.height / 3.0F},
        };
        const std::vector<cv::Point2f> boundary = {
            {0, 0}, {options.width / 2.0F, 0}, {options.width - 1.0F, 0},
            {options.width - 1.0F, options.height / 2.0F},
            {options.width - 1.0F, options.height - 1.0F},
            {options.width / 2.0F, options.height - 1.0F},
            {0, options.height - 1.0F}, {0, options.height / 2.0F},
        };

        std::vector<cv::Mat> normalized_images;
        std::vector<std::vector<cv::Point2f>> normalized_points;
        std::vector<cv::Point2f> average_points(68 + boundary.size(),
                                                cv::Point2f(0, 0));
        for (std::size_t i = 0; i < images.size(); ++i) {
            const std::vector<cv::Point2f> eye_source = {
                point_sets[i][36], point_sets[i][45]};
            const cv::Mat transform = similarityTransform(eye_source, eye_target);
            cv::Mat normalized;
            cv::warpAffine(images[i], normalized, transform, output_size);
            std::vector<cv::Point2f> points;
            cv::transform(point_sets[i], points, transform);
            points.insert(points.end(), boundary.begin(), boundary.end());
            for (std::size_t j = 0; j < points.size(); ++j) {
                average_points[j] +=
                    points[j] * (1.0F / static_cast<float>(images.size()));
            }
            normalized_images.push_back(normalized);
            normalized_points.push_back(points);
        }

        std::vector<cv::Vec3i> triangles;
        calculateDelaunayTriangles(
            cv::Rect(0, 0, options.width, options.height), average_points,
            triangles);
        cv::Mat output = cv::Mat::zeros(output_size, CV_32FC3);
        for (std::size_t i = 0; i < normalized_images.size(); ++i) {
            cv::Mat warped = cv::Mat::zeros(output_size, CV_32FC3);
            for (const auto& triangle : triangles) {
                std::vector<cv::Point2f> source_triangle;
                std::vector<cv::Point2f> target_triangle;
                for (int vertex = 0; vertex < 3; ++vertex) {
                    source_triangle.push_back(constrainPoint(
                        normalized_points[i][triangle[vertex]], output_size));
                    target_triangle.push_back(constrainPoint(
                        average_points[triangle[vertex]], output_size));
                }
                warpTriangle(normalized_images[i], warped, source_triangle,
                             target_triangle);
            }
            output += warped;
        }
        output /= static_cast<double>(images.size());
        cv::Mat output_8bit;
        output.convertTo(output_8bit, CV_8UC3, 255.0);
        if (!options.output.parent_path().empty()) {
            fs::create_directories(options.output.parent_path());
        }
        if (!cv::imwrite(options.output.string(), output_8bit)) {
            throw std::runtime_error("Could not write output image");
        }

        std::cout << "OpenCV: " << CV_VERSION << "\n"
                  << "Inputs: " << images.size() << "\n"
                  << "Delaunay triangles: " << triangles.size() << "\n"
                  << "Saved: " << options.output << "\n";

        if (options.validate) {
            cv::Scalar mean;
            cv::Scalar standard_deviation;
            cv::meanStdDev(output_8bit, mean, standard_deviation);
            const double spread =
                (standard_deviation[0] + standard_deviation[1] +
                 standard_deviation[2]) /
                3.0;
            if (triangles.size() < 50 || spread < 10.0) {
                throw std::runtime_error("Average-face validation failed");
            }
            std::cout << "Validation: PASS\n";
        }
        if (options.display) {
            cv::imshow("Average face", output);
            cv::waitKey(0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
