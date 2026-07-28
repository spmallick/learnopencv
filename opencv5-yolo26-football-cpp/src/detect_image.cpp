// detect_image.cpp
//
// YOLO26 object detection on a single image with the OpenCV 5 DNN module (C++).
// Loads a YOLO26 ONNX model, runs one warm-up pass so the graph is built and
// optimized, times a clean pass, draws the detections, and writes the result.
//
// Usage (run from the repo root; no args needed for a default demo run):
//   detect_image --model models/yolo26n_640.onnx --imgsz 640
//                --source assets/images/team_lineup.jpg --engine auto
//                --out outputs/image_detected.jpg
#include "yolo26_dnn.hpp"
#include "cli.hpp"

#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    const int imgsz          = cli.geti("imgsz", 640);
    const std::string model  = cli.get("model", "models/yolo26n_640.onnx");
    const std::string source = cli.get("source", "assets/images/team_lineup.jpg");
    const std::string engine = cli.get("engine", "auto");
    const float conf         = cli.getf("conf", 0.25f);
    const std::string out    = cli.get("out", "outputs/image_detected.jpg");

    cv::dnn::Net net = y26::build_net(model, engine);

    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "could not read image: " << source << "\n";
        return 1;
    }

    // Warm-up: the first forward pass builds/optimizes the graph.
    y26::detect(net, img, imgsz, conf);

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto dets = y26::detect(net, img, imgsz, conf);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    y26::draw(img, dets);
    y26::ensure_parent_dir(out);
    if (!cv::imwrite(out, img)) {
        std::cerr << "could not write output: " << out << "\n";
        return 1;
    }

    std::map<std::string, int> counts;
    for (const auto& d : dets)
        counts[y26::display_name(d.cls)]++;

    std::cout << "engine=" << engine << "  imgsz=" << imgsz
              << "  inference=" << ms << " ms  detections=" << dets.size() << "\n";
    std::cout << "by class:";
    for (const auto& c : counts) std::cout << "  " << c.first << "=" << c.second;
    std::cout << "\nsaved: " << out << "\n";
    return 0;
}
