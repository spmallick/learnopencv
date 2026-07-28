// yolo26_dnn.hpp
//
// Shared helpers for running YOLO26 with the OpenCV 5 DNN module, in C++.
//
// YOLO26 is an end-to-end, NMS-free detector. Exported to ONNX it emits a fixed
// [1, 300, 6] tensor where each of the 300 rows is:
//
//     [x1, y1, x2, y2, score, class_id]     (box in letterboxed size x size space)
//
// Because the model already selects its final detections, the whole classic
// YOLOv8 post-processing stack (transpose [1,84,8400], objectness, per-class
// NMS) collapses into a single score threshold + a coordinate rescale.
//
// This is a direct port of the Part 1 Python helper (yolo26_dnn.py). The exact
// same three OpenCV calls -- readNetFromONNX / setInput / forward -- carry over
// unchanged, which is the whole point of the C++ follow-up.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace y26 {

// One finalized detection, box in ORIGINAL image pixel coordinates.
struct Detection {
    cv::Rect2f box;   // x1, y1 (top-left) .. x2, y2 (bottom-right)
    float      score;
    int        cls;
};

// The 80 COCO class names (YOLO26 is trained on COCO).
extern const std::vector<std::string> COCO_NAMES;

// Convenience indices we reference a lot on football footage.
constexpr int CLS_PERSON      = 0;
constexpr int CLS_SPORTS_BALL = 32;

// Map a friendly engine string to the OpenCV 5 EngineType and load the model.
//   "auto"    -> cv::dnn::ENGINE_AUTO    (new CPU graph engine, falls back to classic)
//   "new"     -> cv::dnn::ENGINE_NEW     (force the new OpenCV 5 graph engine)
//   "classic" -> cv::dnn::ENGINE_CLASSIC (the 4.x layer engine; needed for CUDA)
cv::dnn::Net build_net(const std::string& onnx_path, const std::string& engine = "auto");

// Resize keeping aspect ratio and pad to a square of `size`. Fills `r` (scale)
// and `dw`,`dh` (padding) so detections can be mapped back to the original.
cv::Mat letterbox(const cv::Mat& img, int size, float& r, int& dw, int& dh,
                  const cv::Scalar& color = cv::Scalar(114, 114, 114));

// Run YOLO26 on a BGR image. Returns detections in ORIGINAL image coordinates.
// No NMS: YOLO26 is NMS-free.
std::vector<Detection> detect(cv::dnn::Net& net, const cv::Mat& img,
                              int size = 640, float conf_thres = 0.25f);

// Stable, distinct BGR colour per class id.
cv::Scalar color_for(int cls);

// Football-friendly label: person -> "player", sports ball -> "ball".
std::string display_name(int cls);

// Resolution-aware drawing sizes: fills box thickness, font scale, font
// thickness so annotations stay readable from a 640 clip up to a 4K still.
void visual_scale(const cv::Mat& img, int& box_th, double& font, int& font_th);

// Draw a filled label chip with bold, auto-contrast text with its top-left at
// `anchor` (usually a box's top-left corner).
void draw_label(cv::Mat& img, const cv::Point& anchor, const std::string& text,
                const cv::Scalar& color, double font, int font_th);

// Draw boxes + "name score" labels on `img` in place (thickness 0 = auto-scale).
void draw(cv::Mat& img, const std::vector<Detection>& dets, int thickness = 0);

// Create the parent directory of `path` if it does not exist, so writing an
// --out like outputs/foo.jpg works on a fresh checkout without a manual mkdir.
void ensure_parent_dir(const std::string& path);

// ---------------------------------------------------------------------------
// Pose (keypoints).  YOLO26-pose ONNX output is [1, 300, 57]:
//   6 detection values [x1,y1,x2,y2,score,class] + 17 keypoints x (x,y,conf).
// ---------------------------------------------------------------------------
struct Keypoint { float x, y, conf; };

struct PoseDetection {
    cv::Rect2f box;
    float score;
    std::vector<Keypoint> kpts;   // 17 COCO keypoints, in ORIGINAL image coords
};

std::vector<PoseDetection> detect_pose(cv::dnn::Net& net, const cv::Mat& img,
                                       int size = 640, float conf_thres = 0.30f);
// skeleton_only = true draws just limbs + joints (no box/label), which is what
// you want when overlaying pose on top of segmentation masks.
void draw_pose(cv::Mat& img, const std::vector<PoseDetection>& poses,
               bool skeleton_only = false, float kpt_thres = 0.30f);

// ---------------------------------------------------------------------------
// Instance segmentation.  YOLO26-seg ONNX has two outputs:
//   output0 [1, 300, 38]  = 6 detection values + 32 mask coefficients
//   output1 [1, 32, 160, 160] = mask prototypes
// ---------------------------------------------------------------------------
struct SegDetection {
    cv::Rect2f box;
    float score;
    int   cls;
    cv::Mat mask;    // 8-bit single-channel, ORIGINAL image size, 0/255
};

std::vector<SegDetection> detect_seg(cv::dnn::Net& net, const cv::Mat& img,
                                     int size = 640, float conf_thres = 0.30f);
void draw_seg(cv::Mat& img, const std::vector<SegDetection>& segs, float alpha = 0.5f);

}  // namespace y26
