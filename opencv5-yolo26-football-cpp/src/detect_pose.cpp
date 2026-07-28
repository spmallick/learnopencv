// detect_pose.cpp
//
// YOLO26-pose keypoint estimation with the OpenCV 5 DNN module (C++). Draws a
// 17-point COCO skeleton on every detected person. Same NMS-free pipeline as the
// detector, the ONNX output is just [1, 300, 57] = 6 box/score/class values plus
// 17 keypoints x (x, y, confidence). Works on an image or a video (auto-detected).
//
// Usage (run from the repo root; no args needed for a default demo run):
//   detect_pose --model models/yolo26n_pose_640.onnx --imgsz 640
//               --source assets/images/aerial_duel.jpg --out outputs/pose.jpg
#include "yolo26_dnn.hpp"
#include "cli.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    const int imgsz          = cli.geti("imgsz", 640);
    const std::string model  = cli.get("model", "models/yolo26n_pose_640.onnx");
    const std::string source = cli.get("source", "assets/images/aerial_duel.jpg");
    const std::string engine = cli.get("engine", "auto");
    const float conf         = cli.getf("conf", 0.30f);
    const float kpt_conf     = cli.getf("kpt-conf", 0.30f);
    const std::string out    = cli.get("out", "outputs/pose.jpg");

    cv::dnn::Net net = y26::build_net(model, engine);
    y26::ensure_parent_dir(out);

    cv::Mat img = cv::imread(source);
    if (!img.empty()) {
        y26::detect_pose(net, img, imgsz, conf);               // warm-up
        auto poses = y26::detect_pose(net, img, imgsz, conf);
        y26::draw_pose(img, poses, false, kpt_conf);
        if (!cv::imwrite(out, img)) {
            std::cerr << "could not write output: " << out << "\n";
            return 1;
        }
        std::cout << "poses=" << poses.size() << "  saved " << out << "\n";
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
        auto poses = y26::detect_pose(net, frame, imgsz, conf);
        y26::draw_pose(frame, poses, false, kpt_conf);
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
