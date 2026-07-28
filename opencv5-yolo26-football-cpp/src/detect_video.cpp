// detect_video.cpp
//
// YOLO26 object detection on a video with the OpenCV 5 DNN module (C++).
// Runs YOLO26 frame by frame, draws the detections, overlays a live FPS banner,
// and writes an annotated .mp4. Optionally restricts to a [--start,--end] second
// window to keep demo clips short.
//
// Usage (run from the repo root; no args needed for a default demo run):
//   detect_video --model models/yolo26n_640.onnx --imgsz 640
//                --source assets/videos/match_play.mp4 --start 0 --end 12
//                --out outputs/video_detected.mp4
#include "yolo26_dnn.hpp"
#include "cli.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utility.hpp>   // cv::getVersionString
#include <chrono>
#include <deque>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    const int imgsz          = cli.geti("imgsz", 640);
    const std::string model  = cli.get("model", "models/yolo26n_640.onnx");
    const std::string source = cli.get("source", "assets/videos/match_play.mp4");
    const std::string engine = cli.get("engine", "auto");
    const float conf         = cli.getf("conf", 0.30f);
    const double start_s     = cli.getf("start", 0.0f);
    const double end_s       = cli.getf("end", -1.0f);
    const std::string out    = cli.get("out", "outputs/video_detected.mp4");

    cv::dnn::Net net = y26::build_net(model, engine);
    y26::ensure_parent_dir(out);

    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr << "could not open video: " << source << "\n";
        return 1;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int start_f = (int)(start_s * fps);
    const int end_f   = end_s > 0 ? (int)(end_s * fps)
                                  : (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    cap.set(cv::CAP_PROP_POS_FRAMES, start_f);

    cv::VideoWriter writer(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(W, H));
    if (!writer.isOpened()) {
        std::cerr << "could not open writer: " << out << "\n";
        return 1;
    }

    const std::string version = cv::getVersionString();
    std::deque<double> fps_hist;
    int idx = start_f;
    double total_ms = 0.0;
    const auto wall0 = std::chrono::high_resolution_clock::now();  // end-to-end timer
    cv::Mat frame;
    while (idx < end_f && cap.read(frame)) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto dets = y26::detect(net, frame, imgsz, conf);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        total_ms += dt * 1000.0;

        fps_hist.push_back(dt > 0 ? 1.0 / dt : 0.0);
        if (fps_hist.size() > 30) fps_hist.pop_front();
        const double live_fps =
            std::accumulate(fps_hist.begin(), fps_hist.end(), 0.0) / fps_hist.size();

        y26::draw(frame, dets);
        char banner[160];
        std::snprintf(banner, sizeof(banner),
                      "OpenCV %s | YOLO26 | engine=%s | %4.1f FPS | %zu obj",
                      version.c_str(), engine.c_str(), live_fps, dets.size());
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(W, 22),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, banner, cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        writer.write(frame);
        ++idx;
    }
    cap.release();
    writer.release();
    const auto wall1 = std::chrono::high_resolution_clock::now();
    const double wall = std::chrono::duration<double>(wall1 - wall0).count();

    const int n = idx - start_f;
    if (n == 0) {
        std::cerr << "video contained no readable frames in the requested range: "
                  << source << "\n";
        return 1;
    }
    std::cout << "processed " << n << " frames | detect avg "
              << (total_ms / std::max(n, 1)) << " ms/frame | detect-only "
              << (1000.0 * n / std::max(total_ms, 1.0)) << " FPS\n";
    std::cout << "end-to-end (decode+detect+draw+encode): " << wall << " s | "
              << (n / std::max(wall, 1e-9)) << " FPS | saved " << out << "\n";
    return 0;
}
