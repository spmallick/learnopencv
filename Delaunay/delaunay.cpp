#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// OpenCV 5 moved Subdiv2D from imgproc into the new geometry module.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// The CMake target supplies the tracked asset directory at compile time.
#ifndef DELAUNAY_SOURCE_DIR
#define DELAUNAY_SOURCE_DIR "."
#endif

// These values form the ordering-independent regression contract for bundled data.
constexpr int kExpectedWidth = 512;
constexpr int kExpectedHeight = 697;
constexpr std::size_t kExpectedPointRecords = 68;
constexpr std::size_t kExpectedUniquePoints = 66;
constexpr std::size_t kExpectedDuplicatePoints = 2;
constexpr std::size_t kExpectedTriangles = 110;
constexpr std::size_t kExpectedUniqueEdges = 175;
constexpr std::size_t kExpectedBoundaryEdges = 20;
constexpr std::size_t kExpectedInteriorEdges = 155;
constexpr std::size_t kExpectedVoronoiFacets = 66;
constexpr double kExpectedTriangleArea = 30853.0;

// Integer point pairs provide standard ordering for sets, maps, and canonical output.
using IntPoint = std::pair<int, int>;
// Sorting all three vertices makes a triangle independent of OpenCV's vertex order.
using Triangle = std::array<IntPoint, 3>;
// Sorting both endpoints makes an edge independent of its traversal direction.
using Edge = std::pair<IntPoint, IntPoint>;

struct Options {
    // Defaults resolve from the source directory instead of the current directory.
    fs::path image_path = fs::path(DELAUNAY_SOURCE_DIR) / "obama.jpg";
    // The bundled two-column landmark file accompanies the default image.
    fs::path points_path = fs::path(DELAUNAY_SOURCE_DIR) / "obama.txt";
    // Generated artifacts stay in an explicit subdirectory by default.
    fs::path output_dir = fs::path(DELAUNAY_SOURCE_DIR) / "output";
    // GUI display is opt-in so servers and continuous integration never block.
    bool display = false;
    // Progressive point insertion is also opt-in and requires display.
    bool animate = false;
    // A short positive delay keeps an animation responsive.
    int animation_delay_ms = 100;
    // Validation activates the bundled-data regression contract.
    bool validate = false;
    // Help prints usage without executing the example.
    bool help = false;
};

struct PointData {
    // Subdiv2D receives only unique sites in stable source order.
    std::vector<cv::Point2f> unique_points;
    // The record count includes exact duplicates skipped before insertion.
    std::size_t record_count = 0;
};

struct VoronoiData {
    // Each facet is a floating-point convex polygon returned by Subdiv2D.
    std::vector<std::vector<cv::Point2f>> facets;
    // Each center is the generating site paired with the corresponding facet.
    std::vector<cv::Point2f> centers;
};

struct GeometryMetrics {
    // The following fields deliberately avoid undocumented collection ordering.
    int width = 0;
    int height = 0;
    std::size_t point_records = 0;
    std::size_t unique_points = 0;
    std::size_t duplicate_points = 0;
    std::size_t triangles = 0;
    std::size_t unique_edges = 0;
    std::size_t boundary_edges = 0;
    std::size_t interior_edges = 0;
    std::size_t voronoi_facets = 0;
    double triangle_area = 0.0;
};

[[nodiscard]] std::string usage(const char* program_name) {
    // Build one concise help string for both --help and parse failures.
    std::ostringstream stream;
    // Name the program and every supported input/output control.
    stream
        << "Usage: " << program_name << " [options]\n"
        << "  --image PATH                 Input image\n"
        << "  --points PATH                Two-column landmark file\n"
        << "  --output-dir PATH            Output directory\n"
        << "  --display | --no-display     Toggle GUI windows (default: off)\n"
        << "  --animate | --no-animation   Toggle insertion animation\n"
        << "  --animation-delay-ms N       Positive animation delay\n"
        << "  --validate                   Check bundled regression data\n"
        << "  --help                       Show this message\n";
    // Return an ordinary string suitable for stdout or an exception message.
    return stream.str();
}

[[nodiscard]] std::string require_value(
    int argc,
    char** argv,
    int& index,
    const std::string& option
) {
    // Every path and integer option needs one following argument.
    if (index + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value");
    }
    // Advance the caller's loop index and return the consumed token.
    return argv[++index];
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    // Start with script-relative defaults matching the Python example.
    Options options;
    // Process each command-line token once from left to right.
    for (int index = 1; index < argc; ++index) {
        // Copy argv data into a managed string before comparisons.
        const std::string argument = argv[index];
        // Override the bundled portrait when requested.
        if (argument == "--image") {
            options.image_path = require_value(argc, argv, index, argument);
        // Override the bundled landmark file when requested.
        } else if (argument == "--points") {
            options.points_path = require_value(argc, argv, index, argument);
        // Select a caller-controlled artifact directory.
        } else if (argument == "--output-dir") {
            options.output_dir = require_value(argc, argv, index, argument);
        // Enable final GUI windows explicitly.
        } else if (argument == "--display") {
            options.display = true;
        // Disable GUI windows explicitly for readable automation commands.
        } else if (argument == "--no-display") {
            options.display = false;
        // Enable progressive triangulation while points are inserted.
        } else if (argument == "--animate") {
            options.animate = true;
        // Disable progressive insertion explicitly.
        } else if (argument == "--no-animation") {
            options.animate = false;
        // Parse a positive delay rather than hardcoding waitKey behavior.
        } else if (argument == "--animation-delay-ms") {
            const std::string value = require_value(argc, argv, index, argument);
            // std::stoi rejects nonnumeric prefixes through its exception behavior.
            std::size_t consumed = 0;
            options.animation_delay_ms = std::stoi(value, &consumed);
            // Reject trailing text such as "100ms".
            if (consumed != value.size()) {
                throw std::invalid_argument(
                    "--animation-delay-ms must be an integer"
                );
            }
        // Activate stable geometric regression checks.
        } else if (argument == "--validate") {
            options.validate = true;
        // Print usage without touching input or output files.
        } else if (argument == "--help" || argument == "-h") {
            options.help = true;
        // Fail on typos rather than silently ignoring unsupported controls.
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    // Invisible animation is almost certainly a command-line mistake.
    if (options.animate && !options.display) {
        throw std::invalid_argument("--animate requires --display");
    }
    // cv::waitKey needs a positive delay during insertion animation.
    if (options.animation_delay_ms <= 0) {
        throw std::invalid_argument(
            "--animation-delay-ms must be positive"
        );
    }
    // Return the fully validated configuration.
    return options;
}

[[nodiscard]] bool rect_contains(
    const cv::Rect& rect,
    const cv::Point2f& point
) {
    // OpenCV rectangles are half-open at the right and bottom boundaries.
    return (
        static_cast<float>(rect.x) <= point.x
        && point.x < static_cast<float>(rect.x + rect.width)
        && static_cast<float>(rect.y) <= point.y
        && point.y < static_cast<float>(rect.y + rect.height)
    );
}

[[nodiscard]] IntPoint rounded_point(const cv::Point2f& point) {
    // cvRound is the conversion used by OpenCV's official Delaunay sample.
    return {cvRound(point.x), cvRound(point.y)};
}

[[nodiscard]] cv::Point to_cv_point(const IntPoint& point) {
    // Convert the canonical standard-library pair back to an OpenCV raster point.
    return {point.first, point.second};
}

[[nodiscard]] cv::Scalar deterministic_color(std::size_t index) {
    // Match the Python example's repeatable blue channel.
    const int blue = 64 + static_cast<int>((37 * index) % 192);
    // Match the Python example's repeatable green channel.
    const int green = 64 + static_cast<int>((17 * index) % 192);
    // Match the Python example's repeatable red channel.
    const int red = 64 + static_cast<int>((29 * index) % 192);
    // cv::Scalar uses blue-green-red order for a BGR image.
    return {
        static_cast<double>(blue),
        static_cast<double>(green),
        static_cast<double>(red)
    };
}

[[nodiscard]] PointData load_points(
    const fs::path& points_path,
    const cv::Rect& rect
) {
    // Open the source explicitly so missing inputs produce a controlled error.
    std::ifstream input(points_path);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Point file not found: " + points_path.string()
        );
    }

    // Build the result in source order while filtering exact duplicates.
    PointData data;
    // Float pairs provide deterministic duplicate detection for parsed coordinates.
    std::set<std::pair<float, float>> seen;
    // Preserve line numbers for actionable parse errors.
    std::size_t line_number = 0;
    // Read one record or comment line at a time.
    std::string raw_line;
    while (std::getline(input, raw_line)) {
        // Increment before validation so diagnostics use one-based line numbers.
        ++line_number;
        // Remove trailing comments using the same rule as the Python implementation.
        const std::size_t comment = raw_line.find('#');
        const std::string line = raw_line.substr(0, comment);
        // Parse exactly two floating-point coordinates from the remaining text.
        std::istringstream stream(line);
        double x = 0.0;
        double y = 0.0;
        // Blank or comment-only lines contain no first coordinate.
        if (!(stream >> x)) {
            // Whitespace-only lines are valid separators.
            stream.clear();
            stream >> std::ws;
            if (stream.eof()) {
                continue;
            }
            throw std::runtime_error(
                points_path.string() + ":" + std::to_string(line_number)
                + ": coordinates must be numeric"
            );
        }
        // A nonblank record must contain its y coordinate.
        if (!(stream >> y)) {
            throw std::runtime_error(
                points_path.string() + ":" + std::to_string(line_number)
                + ": expected two coordinates"
            );
        }
        // Reject any unexpected third field.
        std::string extra;
        if (stream >> extra) {
            throw std::runtime_error(
                points_path.string() + ":" + std::to_string(line_number)
                + ": expected two coordinates"
            );
        }
        // NaN and infinity would otherwise surface as opaque Subdiv2D errors.
        if (!std::isfinite(x) || !std::isfinite(y)) {
            throw std::runtime_error(
                points_path.string() + ":" + std::to_string(line_number)
                + ": coordinates must be finite"
            );
        }
        // This raster tutorial intentionally accepts integer pixel landmarks only.
        if (std::trunc(x) != x || std::trunc(y) != y) {
            throw std::runtime_error(
                points_path.string() + ":" + std::to_string(line_number)
                + ": coordinates must be integers"
            );
        }
        // Convert into the Point2f precision expected by Subdiv2D.
        const cv::Point2f point(
            static_cast<float>(x),
            static_cast<float>(y)
        );
        // Subdiv2D rejects any site outside its initialization rectangle.
        if (!rect_contains(rect, point)) {
            throw std::runtime_error(
                points_path.string() + ":" + std::to_string(line_number)
                + ": point is outside the image rectangle"
            );
        }

        // Count every valid source record, including an exact duplicate.
        ++data.record_count;
        // Insert returns the existing vertex for duplicates; filter them explicitly.
        const auto key = std::make_pair(point.x, point.y);
        if (!seen.insert(key).second) {
            continue;
        }
        // Preserve the first occurrence for repeatable insertion and animation.
        data.unique_points.push_back(point);
    }

    // A triangulation needs at least three distinct sites.
    if (data.unique_points.size() < 3) {
        throw std::runtime_error(
            points_path.string()
            + ": at least three unique points are required"
        );
    }
    // Return both unique sites and the transparent source-record count.
    return data;
}

[[nodiscard]] Triangle canonical_triangle(const cv::Vec6f& raw_triangle) {
    // Convert all three Point2f vertices with the official cvRound rule.
    Triangle triangle = {
        rounded_point({raw_triangle[0], raw_triangle[1]}),
        rounded_point({raw_triangle[2], raw_triangle[3]}),
        rounded_point({raw_triangle[4], raw_triangle[5]})
    };
    // Sorting removes any dependency on clockwise order or starting vertex.
    std::sort(triangle.begin(), triangle.end());
    // Return the canonical integer representation.
    return triangle;
}

[[nodiscard]] std::vector<Triangle> collect_triangles(
    const cv::Subdiv2D& subdiv,
    const cv::Rect& rect
) {
    // Subdiv2D exposes every triangle as six floating-point coordinates.
    std::vector<cv::Vec6f> raw_triangles;
    subdiv.getTriangleList(raw_triangles);
    // A set prevents duplicate triangles and supplies deterministic iteration.
    std::set<Triangle> canonical_triangles;

    // Inspect each returned triangle independently of collection ordering.
    for (const cv::Vec6f& raw_triangle : raw_triangles) {
        // Retain floating-point coordinates for the half-open bounds test.
        const std::array<cv::Point2f, 3> vertices = {
            cv::Point2f(raw_triangle[0], raw_triangle[1]),
            cv::Point2f(raw_triangle[2], raw_triangle[3]),
            cv::Point2f(raw_triangle[4], raw_triangle[5])
        };
        // Ignore any triangle that is not completely inside the image rectangle.
        if (
            !rect_contains(rect, vertices[0])
            || !rect_contains(rect, vertices[1])
            || !rect_contains(rect, vertices[2])
        ) {
            continue;
        }
        // Canonicalization makes geometry checks ordering-independent.
        canonical_triangles.insert(canonical_triangle(raw_triangle));
    }

    // Convert the sorted set into the sequence used by drawing and validation.
    return {
        canonical_triangles.begin(),
        canonical_triangles.end()
    };
}

[[nodiscard]] VoronoiData collect_voronoi(cv::Subdiv2D& subdiv) {
    // An empty vertex-ID list requests every nonvirtual Voronoi facet.
    const std::vector<int> all_vertices;
    // Receive paired floating-point facets and their generating sites.
    VoronoiData data;
    subdiv.getVoronoiFacetList(
        all_vertices,
        data.facets,
        data.centers
    );
    // A mismatch would make cell-to-site rendering ambiguous.
    if (data.facets.size() != data.centers.size()) {
        throw std::runtime_error(
            "OpenCV returned mismatched Voronoi facets and centers"
        );
    }
    // Build a stable cell order from the unique integer generating sites.
    std::vector<std::pair<IntPoint, std::size_t>> cell_order;
    cell_order.reserve(data.centers.size());
    std::set<IntPoint> unique_centers;
    for (std::size_t index = 0; index < data.centers.size(); ++index) {
        const IntPoint center = rounded_point(data.centers[index]);
        if (!unique_centers.insert(center).second) {
            throw std::runtime_error(
                "OpenCV returned duplicate rounded Voronoi centers"
            );
        }
        cell_order.emplace_back(center, index);
    }
    // Palette assignment must not depend on the API's facet collection order.
    std::sort(cell_order.begin(), cell_order.end());

    // Rebuild the paired sequences in canonical center order.
    VoronoiData sorted_data;
    sorted_data.facets.reserve(data.facets.size());
    sorted_data.centers.reserve(data.centers.size());
    for (const auto& [center, original_index] : cell_order) {
        // The center key controls ordering; the paired Point2f value is preserved.
        static_cast<void>(center);
        sorted_data.facets.push_back(data.facets[original_index]);
        sorted_data.centers.push_back(data.centers[original_index]);
    }
    // Return deterministic cells shared with the Python implementation.
    return sorted_data;
}

void draw_point(
    cv::Mat& image,
    const cv::Point2f& point,
    const cv::Scalar& color
) {
    // Convert Point2f explicitly because raster drawing uses integer coordinates.
    const cv::Point center = to_cv_point(rounded_point(point));
    // FILLED creates a solid marker and LINE_AA keeps its edge smooth.
    cv::circle(
        image,
        center,
        2,
        color,
        cv::FILLED,
        cv::LINE_AA
    );
}

void draw_delaunay(
    cv::Mat& image,
    const std::vector<Triangle>& triangles,
    const cv::Scalar& color
) {
    // Draw all three edges of every canonical triangle.
    for (const Triangle& triangle : triangles) {
        // Convert the canonical pair representation into OpenCV points once.
        const cv::Point point_1 = to_cv_point(triangle[0]);
        const cv::Point point_2 = to_cv_point(triangle[1]);
        const cv::Point point_3 = to_cv_point(triangle[2]);
        // Draw the first one-pixel antialiased edge.
        cv::line(image, point_1, point_2, color, 1, cv::LINE_AA);
        // Draw the second edge using identical raster settings.
        cv::line(image, point_2, point_3, color, 1, cv::LINE_AA);
        // Close the triangle with its final edge.
        cv::line(image, point_3, point_1, color, 1, cv::LINE_AA);
    }
}

void draw_voronoi(cv::Mat& image, const VoronoiData& data) {
    // Draw each facet with the same deterministic palette as the Python example.
    for (std::size_t index = 0; index < data.facets.size(); ++index) {
        // A valid Voronoi polygon needs at least three vertices.
        if (data.facets[index].size() < 3) {
            throw std::runtime_error(
                "Voronoi facet " + std::to_string(index)
                + " has fewer than 3 vertices"
            );
        }
        // Convert Point2f polygon vertices explicitly before rasterization.
        std::vector<cv::Point> polygon;
        polygon.reserve(data.facets[index].size());
        for (const cv::Point2f& vertex : data.facets[index]) {
            polygon.push_back(to_cv_point(rounded_point(vertex)));
        }
        // Fill the cell before drawing its outline and center marker.
        cv::fillConvexPoly(
            image,
            polygon,
            deterministic_color(index),
            cv::LINE_AA
        );
        // polylines accepts a sequence of contours; this call has one polygon.
        const std::vector<std::vector<cv::Point>> polygons = {polygon};
        // A black outline separates adjacent cells with similar colors.
        cv::polylines(
            image,
            polygons,
            true,
            cv::Scalar(0, 0, 0),
            1,
            cv::LINE_AA
        );
        // Mark the generating site paired with this facet.
        cv::circle(
            image,
            to_cv_point(rounded_point(data.centers[index])),
            3,
            cv::Scalar(0, 0, 0),
            cv::FILLED,
            cv::LINE_AA
        );
    }
}

[[nodiscard]] double triangle_area(const Triangle& triangle) {
    // Promote integer products to double before applying the shoelace formula.
    const double x_1 = triangle[0].first;
    const double y_1 = triangle[0].second;
    const double x_2 = triangle[1].first;
    const double y_2 = triangle[1].second;
    const double x_3 = triangle[2].first;
    const double y_3 = triangle[2].second;
    // The determinant below is twice the signed area.
    const double twice_area = (
        x_1 * (y_2 - y_3)
        + x_2 * (y_3 - y_1)
        + x_3 * (y_1 - y_2)
    );
    // Regression checks use unsigned pixel-coordinate area.
    return std::abs(twice_area) / 2.0;
}

[[nodiscard]] Edge normalized_edge(
    const IntPoint& first,
    const IntPoint& second
) {
    // Store the lexicographically smaller endpoint first.
    return first < second
        ? Edge{first, second}
        : Edge{second, first};
}

[[nodiscard]] std::array<std::size_t, 3> edge_counts(
    const std::vector<Triangle>& triangles
) {
    // Map each undirected edge to its number of incident triangles.
    std::map<Edge, std::size_t> incidences;
    // Visit every triangle independently of its canonical vertex order.
    for (const Triangle& triangle : triangles) {
        // Increment the first normalized edge.
        ++incidences[normalized_edge(triangle[0], triangle[1])];
        // Increment the second normalized edge.
        ++incidences[normalized_edge(triangle[1], triangle[2])];
        // Increment the closing normalized edge.
        ++incidences[normalized_edge(triangle[2], triangle[0])];
    }

    // Hull edges have one incident triangle.
    std::size_t boundary_edges = 0;
    // Interior edges have two incident triangles.
    std::size_t interior_edges = 0;
    // Check every edge for manifold triangulation behavior.
    for (const auto& [edge, count] : incidences) {
        // The edge key itself is useful for ordering but not needed in this count.
        static_cast<void>(edge);
        if (count == 1) {
            ++boundary_edges;
        } else if (count == 2) {
            ++interior_edges;
        } else {
            throw std::runtime_error(
                "Triangulation contains an invalid edge incidence"
            );
        }
    }
    // Return total, boundary, and interior counts together.
    return {incidences.size(), boundary_edges, interior_edges};
}

[[nodiscard]] GeometryMetrics calculate_metrics(
    const cv::Mat& image,
    const PointData& points,
    const std::vector<Triangle>& triangles,
    const VoronoiData& voronoi
) {
    // Count topology without relying on OpenCV triangle ordering.
    const auto counts = edge_counts(triangles);
    // Sum positive areas for an ordering-independent geometric measurement.
    double total_area = 0.0;
    for (const Triangle& triangle : triangles) {
        const double area = triangle_area(triangle);
        total_area += area;
    }

    // Return one immutable collection of stable metrics.
    return {
        image.cols,
        image.rows,
        points.record_count,
        points.unique_points.size(),
        points.record_count - points.unique_points.size(),
        triangles.size(),
        counts[0],
        counts[1],
        counts[2],
        voronoi.facets.size(),
        total_area
    };
}

void validate_metrics(
    const GeometryMetrics& metrics,
    const PointData& points,
    const std::vector<Triangle>& triangles,
    const VoronoiData& voronoi
) {
    // Rounded membership checks are appropriate for the bundled integer landmarks.
    std::set<IntPoint> rounded_sites;
    for (const cv::Point2f& point : points.unique_points) {
        rounded_sites.insert(rounded_point(point));
    }
    // Canonical Voronoi centers should equal the unique bundled sites.
    std::set<IntPoint> rounded_centers;
    for (const cv::Point2f& center : voronoi.centers) {
        rounded_centers.insert(rounded_point(center));
    }
    if (rounded_centers != rounded_sites) {
        throw std::runtime_error(
            "Validation failed: Voronoi centers differ from sites"
        );
    }
    // Every Delaunay vertex should also correspond to a bundled site.
    for (const Triangle& triangle : triangles) {
        if (triangle_area(triangle) <= 0.0) {
            throw std::runtime_error(
                "Validation failed: triangle has nonpositive area"
            );
        }
        for (const IntPoint& vertex : triangle) {
            if (rounded_sites.count(vertex) == 0) {
                throw std::runtime_error(
                    "Validation failed: triangle has a non-site vertex"
                );
            }
        }
    }

    // Compare each invariant separately so a failure identifies its cause.
    if (metrics.width != kExpectedWidth) {
        throw std::runtime_error("Validation failed: image width");
    }
    if (metrics.height != kExpectedHeight) {
        throw std::runtime_error("Validation failed: image height");
    }
    if (metrics.point_records != kExpectedPointRecords) {
        throw std::runtime_error("Validation failed: point record count");
    }
    if (metrics.unique_points != kExpectedUniquePoints) {
        throw std::runtime_error("Validation failed: unique point count");
    }
    if (metrics.duplicate_points != kExpectedDuplicatePoints) {
        throw std::runtime_error("Validation failed: duplicate point count");
    }
    if (metrics.triangles != kExpectedTriangles) {
        throw std::runtime_error("Validation failed: triangle count");
    }
    if (metrics.unique_edges != kExpectedUniqueEdges) {
        throw std::runtime_error("Validation failed: unique edge count");
    }
    if (metrics.boundary_edges != kExpectedBoundaryEdges) {
        throw std::runtime_error("Validation failed: boundary edge count");
    }
    if (metrics.interior_edges != kExpectedInteriorEdges) {
        throw std::runtime_error("Validation failed: interior edge count");
    }
    if (metrics.voronoi_facets != kExpectedVoronoiFacets) {
        throw std::runtime_error("Validation failed: Voronoi facet count");
    }
    if (std::abs(metrics.triangle_area - kExpectedTriangleArea) > 1e-9) {
        throw std::runtime_error("Validation failed: triangle area");
    }
    // Euler's formula supplies an independent planar topology check.
    const std::size_t faces_including_exterior = metrics.triangles + 1;
    const auto euler_characteristic =
        static_cast<long long>(metrics.unique_points)
        - static_cast<long long>(metrics.unique_edges)
        + static_cast<long long>(faces_including_exterior);
    if (euler_characteristic != 2) {
        throw std::runtime_error(
            "Validation failed: Euler characteristic is not 2"
        );
    }
}

void validate_rendered_images(
    const cv::Mat& source,
    const cv::Mat& delaunay_image,
    const cv::Mat& voronoi_image
) {
    // Removing all triangle/point drawing must make the regression fail.
    cv::Mat delaunay_difference;
    cv::absdiff(source, delaunay_image, delaunay_difference);
    if (cv::countNonZero(delaunay_difference.reshape(1)) == 0) {
        throw std::runtime_error(
            "Validation failed: Delaunay output equals the input"
        );
    }
    // A blank Voronoi canvas indicates that facet drawing did not run.
    if (cv::countNonZero(voronoi_image.reshape(1)) == 0) {
        throw std::runtime_error(
            "Validation failed: Voronoi output is blank"
        );
    }
    // Nonzero channel variation proves the output is not one flat color.
    cv::Mat mean;
    cv::Mat standard_deviation;
    cv::meanStdDev(voronoi_image, mean, standard_deviation);
    if (cv::norm(standard_deviation) <= 0.0) {
        throw std::runtime_error(
            "Validation failed: Voronoi output lacks color variation"
        );
    }
}

void write_image(const fs::path& path, const cv::Mat& image) {
    // OpenCV reports several encoder and filesystem failures through a bool.
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error(
            "Could not write output image: " + path.string()
        );
    }
    // Decode the artifact again so validation covers the actual file on disk.
    const cv::Mat decoded = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (decoded.empty()) {
        throw std::runtime_error(
            "Could not read generated output image: " + path.string()
        );
    }
    // The generated file must preserve dimensions and three-channel type.
    if (decoded.size() != image.size() || decoded.type() != image.type()) {
        throw std::runtime_error(
            "Generated output shape/type mismatch: " + path.string()
        );
    }
}

[[nodiscard]] fs::path comparable_path(const fs::path& path) {
    // Build an absolute path before resolving existing symlink components.
    std::error_code absolute_error;
    const fs::path absolute = fs::absolute(path, absolute_error);
    if (absolute_error) {
        throw std::runtime_error(
            "Could not resolve path "
            + path.string()
            + ": "
            + absolute_error.message()
        );
    }
    // weakly_canonical also supports output filenames that do not exist yet.
    std::error_code canonical_error;
    const fs::path canonical = fs::weakly_canonical(
        absolute,
        canonical_error
    );
    if (canonical_error) {
        throw std::runtime_error(
            "Could not normalize path "
            + path.string()
            + ": "
            + canonical_error.message()
        );
    }
    // Return one normalized representation for safe equality comparisons.
    return canonical;
}

void ensure_output_paths_are_safe(const Options& options) {
    // Retain original paths for filesystem-equivalence checks on existing files.
    const std::array<fs::path, 2> input_paths = {
        options.image_path,
        options.points_path
    };
    // Check both documented outputs before creating a directory or writing a file.
    const std::array<fs::path, 2> output_paths = {
        options.output_dir / "delaunay.png",
        options.output_dir / "voronoi.png"
    };
    // Refuse to overwrite either the image or the landmark source.
    for (const fs::path& output_path : output_paths) {
        for (const fs::path& input_path : input_paths) {
            // Normalized equality handles ordinary and symlinked path aliases.
            bool paths_match = (
                comparable_path(output_path)
                == comparable_path(input_path)
            );
            // Existing hard links and case aliases require inode equivalence.
            std::error_code output_exists_error;
            const bool output_exists = fs::exists(
                output_path,
                output_exists_error
            );
            if (output_exists_error) {
                throw std::runtime_error(
                    "Could not inspect output path "
                    + output_path.string()
                    + ": "
                    + output_exists_error.message()
                );
            }
            std::error_code input_exists_error;
            const bool input_exists = fs::exists(
                input_path,
                input_exists_error
            );
            if (input_exists_error) {
                throw std::runtime_error(
                    "Could not inspect input path "
                    + input_path.string()
                    + ": "
                    + input_exists_error.message()
                );
            }
            if (!paths_match && output_exists && input_exists) {
                std::error_code equivalent_error;
                paths_match = fs::equivalent(
                    output_path,
                    input_path,
                    equivalent_error
                );
                if (equivalent_error) {
                    throw std::runtime_error(
                        "Could not compare input and output paths: "
                        + equivalent_error.message()
                    );
                }
            }
            if (paths_match) {
                throw std::runtime_error(
                    "Output path would overwrite an input file: "
                    + output_path.string()
                );
            }
        }
    }
}

[[nodiscard]] cv::Subdiv2D build_subdivision(
    const cv::Rect& rect,
    const std::vector<cv::Point2f>& points,
    const cv::Mat& image,
    const Options& options
) {
    // Initialize Subdiv2D over the exact image rectangle.
    cv::Subdiv2D subdiv(rect);
    // Insert every unique point once in its source order.
    for (const cv::Point2f& point : points) {
        // Duplicate records were filtered before this API call.
        subdiv.insert(point);
        // Animation is opt-in so automated and remote runs remain headless.
        if (options.display && options.animate) {
            // Render the partial triangulation on a fresh image copy.
            cv::Mat frame = image.clone();
            // Collection canonicalization avoids depending on API ordering.
            const std::vector<Triangle> partial_triangles =
                collect_triangles(subdiv, rect);
            // White edges remain visible against the portrait.
            draw_delaunay(
                frame,
                partial_triangles,
                cv::Scalar(255, 255, 255)
            );
            // Display the progressive tutorial frame.
            cv::imshow("Delaunay Triangulation", frame);
            // Let the GUI process events between insertions.
            cv::waitKey(options.animation_delay_ms);
        }
    }
    // Return the completed subdivision for final geometry extraction.
    return subdiv;
}

[[nodiscard]] GeometryMetrics run(const Options& options) {
    // Resolve every intended artifact before reading or writing user-controlled paths.
    ensure_output_paths_are_safe(options);
    // Decode the source in color mode and fail before dereferencing an empty Mat.
    const cv::Mat source = cv::imread(
        options.image_path.string(),
        cv::IMREAD_COLOR
    );
    if (source.empty()) {
        throw std::runtime_error(
            "Could not read input image: " + options.image_path.string()
        );
    }
    // This visualization expects an ordinary three-channel BGR image.
    if (source.type() != CV_8UC3) {
        throw std::runtime_error(
            "Expected an 8-bit three-channel input image"
        );
    }

    // Build the half-open Subdiv2D rectangle from decoded dimensions.
    const cv::Rect rect(0, 0, source.cols, source.rows);
    // Parse, validate, and explicitly de-duplicate landmark records.
    const PointData point_data = load_points(options.points_path, rect);
    // Insert sites and optionally render the progressive triangulation.
    cv::Subdiv2D subdiv = build_subdivision(
        rect,
        point_data.unique_points,
        source,
        options
    );
    // Extract canonical triangles after all sites have been inserted.
    const std::vector<Triangle> triangles =
        collect_triangles(subdiv, rect);
    // Extract every nonvirtual Voronoi facet and its site.
    const VoronoiData voronoi = collect_voronoi(subdiv);

    // Draw the final triangulation on a copy of the original image.
    cv::Mat delaunay_image = source.clone();
    draw_delaunay(
        delaunay_image,
        triangles,
        cv::Scalar(255, 255, 255)
    );
    // Overlay every unique landmark in red.
    for (const cv::Point2f& point : point_data.unique_points) {
        draw_point(
            delaunay_image,
            point,
            cv::Scalar(0, 0, 255)
        );
    }

    // Start the Voronoi visualization from a black canvas of matching type.
    cv::Mat voronoi_image = cv::Mat::zeros(
        source.rows,
        source.cols,
        CV_8UC3
    );
    // Fill, outline, and mark every facet deterministically.
    draw_voronoi(voronoi_image, voronoi);

    // Calculate order-independent geometry before output or display.
    const GeometryMetrics metrics = calculate_metrics(
        source,
        point_data,
        triangles,
        voronoi
    );
    // Activate the bundled-data acceptance contract only when requested.
    if (options.validate) {
        validate_metrics(metrics, point_data, triangles, voronoi);
        validate_rendered_images(
            source,
            delaunay_image,
            voronoi_image
        );
    }

    // Create every missing parent directory and report filesystem errors.
    std::error_code directory_error;
    fs::create_directories(options.output_dir, directory_error);
    if (directory_error) {
        throw std::runtime_error(
            "Could not create output directory "
            + options.output_dir.string()
            + ": "
            + directory_error.message()
        );
    }
    // Lossless PNG avoids codec-dependent JPEG artifacts.
    write_image(
        options.output_dir / "delaunay.png",
        delaunay_image
    );
    // Verify the Voronoi artifact through the same write/read path.
    write_image(
        options.output_dir / "voronoi.png",
        voronoi_image
    );

    // GUI windows remain optional for reliable server and CI behavior.
    if (options.display) {
        // Show the completed triangulation.
        cv::imshow("Delaunay Triangulation", delaunay_image);
        // Show the dual Voronoi diagram.
        cv::imshow("Voronoi Diagram", voronoi_image);
        // Wait for one key before closing both windows.
        cv::waitKey(0);
        // Release GUI resources explicitly.
        cv::destroyAllWindows();
    }

    // Return the metrics for concise CLI reporting.
    return metrics;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // Parse all controls before reading inputs or creating output paths.
        const Options options = parse_options(argc, argv);
        // Help is a successful, side-effect-free command.
        if (options.help) {
            std::cout << usage(argv[0]);
            return 0;
        }

        // Run the complete real entry point and collect validated metrics.
        const GeometryMetrics metrics = run(options);
        // Report the exact linked OpenCV version before geometry results.
        std::cout << "OpenCV version: " << CV_VERSION << '\n';
        // Keep the summary compact enough for readers and CTest logs.
        std::cout
            << "Geometry: "
            << metrics.point_records << " records, "
            << metrics.unique_points << " unique points, "
            << metrics.triangles << " triangles, "
            << metrics.voronoi_facets << " Voronoi facets\n";
        // Emit the success marker only after outputs and metrics pass.
        if (options.validate) {
            std::cout << "DELAUNAY_VALIDATION_OK\n";
        }
        // Zero signals successful rendering and optional validation.
        return 0;
    } catch (const std::exception& error) {
        // Convert expected parse, input, output, and OpenCV errors into one message.
        std::cerr << "error: " << error.what() << '\n';
        // Show available controls after a parse or runtime failure.
        std::cerr << usage(argv[0]);
        // Two denotes invalid input or an unusable runtime environment.
        return 2;
    }
}
