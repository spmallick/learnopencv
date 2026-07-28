// detect_seg.cpp
//
// YOLO26-seg instance segmentation with the OpenCV 5 DNN module (C++). Overlays
// a coloured mask per detected object and outlines it. The seg ONNX has two
// outputs: output0 [1, 300, 38] = 6 detection values + 32 mask coefficients, and
// output1 [1, 32, 160, 160] = mask prototypes. Each object's mask is the sigmoid
// of (coefficients . prototypes), rescaled from proto space back to the frame.
// Works on an image or a video (auto-detected).
//
// Usage (run from the repo root; no args needed for a default demo run):
//   detect_seg --model models/yolo26n_seg_640.onnx --imgsz 640
//              --source assets/images/goal_celebration.jpg --out outputs/seg.jpg
#include "yolo26_dnn.hpp"
#include "cli.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    const int imgsz          = cli.geti("imgsz", 640);
    const std::string model  = cli.get("model", "models/yolo26n_seg_640.onnx");
    const std::string source = cli.get("source", "assets/images/goal_celebration.jpg");
    const std::string engine = cli.get("engine", "auto");
    const float conf         = cli.getf("conf", 0.30f);
    const float alpha        = cli.getf("alpha", 0.5f);
    const std::string out    = cli.get("out", "outputs/seg.jpg");
    if (alpha < 0.0f || alpha > 1.0f) {
        std::cerr << "--alpha must be between 0 and 1\n";
        return 1;
    }

    cv::dnn::Net net = y26::build_net(model, engine);
    y26::ensure_parent_dir(out);

    cv::Mat img = cv::imread(source);
    if (!img.empty()) {
        y26::detect_seg(net, img, imgsz, conf);              // warm-up
        auto segs = y26::detect_seg(net, img, imgsz, conf);
        y26::draw_seg(img, segs, alpha);
        if (!cv::imwrite(out, img)) {
            std::cerr << "could not write output: " << out << "\n";
            return 1;
        }
        std::cout << "instances=" << segs.size() << "  saved " << out << "\n";
        return 0;
    }

    cv::VideoCapture cap(source);
    if (!cap.isOpened()) { std::cerr << "cannot open " << source << "\n"; return 1; }
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer(out, cv::VideoWriter::fourcc('m','p','4','v'), fps, {W, H});
    if (!writer.isOpened()) {
        std::cerr << "could not open writer: " << out << "\n";
        return 1;
    }

    cv::Mat frame; int n = 0;
    while (cap.read(frame)) {
        auto segs = y26::detect_seg(net, frame, imgsz, conf);
        y26::draw_seg(frame, segs, alpha);
        writer.write(frame);
        ++n;
    }
    if (n == 0) {
        std::cerr << "video contained no readable frames: " << source << "\n";
        return 1;
    }
    std::cout << "processed " << n << " frames, saved " << out << "\n";
    return 0;
}
