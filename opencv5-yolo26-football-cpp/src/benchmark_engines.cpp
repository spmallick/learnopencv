// benchmark_engines.cpp
//
// Time the exact same YOLO26 ONNX model + image through each OpenCV 5 DNN engine
// (ENGINE_NEW, ENGINE_CLASSIC, ENGINE_AUTO), averaged over N runs after a warm-up.
// This is the C++ counterpart of the Part 1 Python benchmark and lets us compare
// the new graph engine against the classic layer engine, in C++.
//
// Usage (run from the repo root; no args needed for a default demo run):
//   benchmark_engines --model models/yolo26n_640.onnx --imgsz 640
//                     --source assets/images/team_lineup.jpg --runs 40
#include "yolo26_dnn.hpp"
#include "cli.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

static double bench_one(const std::string& model, const std::string& engine,
                        const cv::Mat& img, int imgsz, int runs) {
    cv::dnn::Net net = y26::build_net(model, engine);
    // A couple of warm-up passes so the graph is built/optimized and caches are hot.
    y26::detect(net, img, imgsz, 0.25f);
    y26::detect(net, img, imgsz, 0.25f);

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i)
        y26::detect(net, img, imgsz, 0.25f);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
}

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    const int imgsz          = cli.geti("imgsz", 640);
    const std::string model  = cli.get("model", "models/yolo26n_640.onnx");
    const std::string source = cli.get("source", "assets/images/team_lineup.jpg");
    const int runs           = cli.geti("runs", 40);
    if (runs <= 0) {
        std::cerr << "--runs must be positive\n";
        return 1;
    }

    cv::Mat img = cv::imread(source);
    if (img.empty()) {
        std::cerr << "could not read image: " << source << "\n";
        return 1;
    }

    std::cout << "OpenCV " << cv::getVersionString()
              << " | model=" << model << " | " << img.cols << "x" << img.rows
              << " | runs=" << runs << "\n";
    std::cout << "-------------------------------------------------\n";
    std::cout << "engine     mean(ms)      FPS\n";

    for (const std::string engine : {"new", "classic", "auto"}) {
        try {
            const double ms = bench_one(model, engine, img, imgsz, runs);
            std::printf("%-9s  %8.1f  %7.1f\n", engine.c_str(), ms, 1000.0 / ms);
        } catch (const std::exception& e) {
            std::printf("%-9s  (failed: %s)\n", engine.c_str(), e.what());
        }
    }
    return 0;
}
