// yolo26_dnn.cpp -- implementation of the shared YOLO26 + OpenCV 5 DNN helpers.
#include "yolo26_dnn.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace y26 {

const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
};

cv::dnn::Net build_net(const std::string& onnx_path, const std::string& engine) {
    int eng = cv::dnn::ENGINE_AUTO;
    std::string e = engine;
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (e == "auto")         eng = cv::dnn::ENGINE_AUTO;
    else if (e == "new")     eng = cv::dnn::ENGINE_NEW;
    else if (e == "classic") eng = cv::dnn::ENGINE_CLASSIC;
    else throw std::invalid_argument("engine must be auto|new|classic, got: " + engine);

    // In OpenCV 5, readNetFromONNX takes the engine as a second argument.
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path, eng);
    if (net.empty())
        throw std::runtime_error("failed to load ONNX model: " + onnx_path);
    return net;
}

void ensure_parent_dir(const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());
}

cv::Mat letterbox(const cv::Mat& img, int size, float& r, int& dw, int& dh,
                  const cv::Scalar& color) {
    if (img.empty())
        throw std::invalid_argument("letterbox input image must not be empty");
    if (size <= 0)
        throw std::invalid_argument("letterbox size must be positive");

    const int h = img.rows, w = img.cols;
    r = std::min(static_cast<float>(size) / h, static_cast<float>(size) / w);
    const int nw = static_cast<int>(std::round(w * r));
    const int nh = static_cast<int>(std::round(h * r));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(size, size, img.type(), color);
    dw = (size - nw) / 2;
    dh = (size - nh) / 2;
    resized.copyTo(canvas(cv::Rect(dw, dh, nw, nh)));
    return canvas;
}

std::vector<Detection> detect(cv::dnn::Net& net, const cv::Mat& img,
                              int size, float conf_thres) {
    float r; int dw, dh;
    cv::Mat padded = letterbox(img, size, r, dw, dh);

    // 1/255 scale, swapRB=true (YOLO expects RGB), crop=false.
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0,
                                          cv::Size(size, size), cv::Scalar(),
                                          /*swapRB=*/true, /*crop=*/false);
    net.setInput(blob);
    cv::Mat out = net.forward();          // shape [1, 300, 6]
    if (out.dims != 3 || out.type() != CV_32F || out.size[2] < 6)
        throw std::runtime_error("unexpected detection output; expected [1,N,6] float tensor");
    if (!out.isContinuous())
        out = out.clone();

    // The output is [1, N, 6]; read N (rows) and 6 (cols) from the shape so the
    // parser does not care whether N is 300 or something else.
    const int rows = out.size[out.dims - 2];
    const int cols = out.size[out.dims - 1];
    const float* p = reinterpret_cast<const float*>(out.data);

    std::vector<Detection> dets;
    dets.reserve(rows);
    for (int i = 0; i < rows; ++i) {
        const float* row = p + static_cast<size_t>(i) * cols;
        const float score = row[4];
        if (score < conf_thres) continue;

        // Undo the letterbox: subtract pad, divide by ratio.
        const float x1 = (row[0] - dw) / r;
        const float y1 = (row[1] - dh) / r;
        const float x2 = (row[2] - dw) / r;
        const float y2 = (row[3] - dh) / r;

        Detection d;
        d.box   = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        d.score = score;
        d.cls   = static_cast<int>(row[5]);
        dets.push_back(d);
    }
    return dets;
}

// Football-friendly display names: COCO calls it "person", but on match footage
// "player" reads better. "sports ball" is shortened to "ball". The person label
// can be overridden with the Y26_PERSON_LABEL env var (e.g. "dancer" for the
// halftime performance clip) without touching any other class.
std::string display_name(int cls) {
    if (cls == CLS_PERSON) {
        const char* ov = std::getenv("Y26_PERSON_LABEL");
        return (ov && *ov) ? std::string(ov) : std::string("player");
    }
    if (cls == CLS_SPORTS_BALL) return "ball";
    if (cls >= 0 && cls < (int)COCO_NAMES.size()) return COCO_NAMES[cls];
    return std::to_string(cls);
}

cv::Scalar color_for(int cls) {
    // A curated, vivid, high-saturation palette (BGR). Distinct and punchy so
    // boxes read clearly on grass without looking washed out.
    static const cv::Scalar PALETTE[] = {
        {56, 56, 255},   {51, 160, 255},  {51, 221, 255},  {51, 255, 160},
        {255, 191, 0},   {255, 128, 0},   {255, 64, 128},  {204, 51, 255},
        {255, 51, 153},  {0, 204, 102},   {0, 165, 255},   {80, 208, 146},
        {255, 102, 178}, {102, 255, 102}, {0, 100, 255},   {255, 153, 51},
        {153, 51, 255},  {51, 255, 221},  {255, 51, 51},   {51, 255, 51},
    };
    const int n = (int)(sizeof(PALETTE) / sizeof(PALETTE[0]));
    return PALETTE[((cls % n) + n) % n];
}

// Pick black or white text for readability on a given fill colour.
static cv::Scalar text_on(const cv::Scalar& bg) {
    const double lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2];  // BGR
    return lum > 150 ? cv::Scalar(20, 20, 20) : cv::Scalar(255, 255, 255);
}

// Draw a filled label chip with a bold, high-contrast caption above a box.
void draw_label(cv::Mat& img, const cv::Point& anchor, const std::string& text,
                const cv::Scalar& color, double font, int font_th) {
    int bl = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, font, font_th, &bl);
    int x = anchor.x, y = anchor.y;
    const int pad = std::max(2, (int)std::lround(font * 4));
    cv::Rect chip(x, y - ts.height - 2 * pad, ts.width + 2 * pad, ts.height + 2 * pad);
    chip.y = std::max(chip.y, 0);
    cv::rectangle(img, chip, color, cv::FILLED, cv::LINE_AA);
    cv::putText(img, text, cv::Point(chip.x + pad, chip.y + ts.height + pad - 1),
                cv::FONT_HERSHEY_DUPLEX, font, text_on(color), font_th, cv::LINE_AA);
}

// Scale line thickness and label size with the image so boxes and text stay
// readable on everything from a 640 clip to a 4K still.
void visual_scale(const cv::Mat& img, int& box_th, double& font, int& font_th) {
    const double s = std::clamp(img.rows / 1500.0, 0.7, 2.2);
    box_th  = std::max(2, (int)std::lround(1.7 * s));   // vivid but not chunky
    font    = std::clamp(0.58 * s, 0.5, 1.15);
    font_th = std::max(1, (int)std::lround(font * 1.5));
}

void draw(cv::Mat& img, const std::vector<Detection>& dets, int thickness) {
    int box_th, font_th; double font;
    visual_scale(img, box_th, font, font_th);
    if (thickness > 0) box_th = thickness;

    for (const auto& d : dets) {
        const cv::Scalar color = color_for(d.cls);
        const cv::Point p1(cvRound(d.box.x), cvRound(d.box.y));
        const cv::Point p2(cvRound(d.box.x + d.box.width),
                           cvRound(d.box.y + d.box.height));
        cv::rectangle(img, p1, p2, color, box_th, cv::LINE_AA);

        const std::string name = display_name(d.cls);
        char label[128];
        std::snprintf(label, sizeof(label), "%s %.2f", name.c_str(), d.score);
        draw_label(img, p1, label, color, font, font_th);
    }
}

// ---------------------------------------------------------------------------
// Pose
// ---------------------------------------------------------------------------
static const int SKELETON[][2] = {
    {15,13},{13,11},{16,14},{14,12},{11,12},{5,11},{6,12},{5,6},{5,7},{6,8},
    {7,9},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6},
};

std::vector<PoseDetection> detect_pose(cv::dnn::Net& net, const cv::Mat& img,
                                       int size, float conf_thres) {
    float r; int dw, dh;
    cv::Mat padded = letterbox(img, size, r, dw, dh);
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(size, size),
                                          cv::Scalar(), true, false);
    net.setInput(blob);
    cv::Mat out = net.forward();            // [1, 300, 57]
    if (out.dims != 3 || out.type() != CV_32F || out.size[2] < 9 ||
        (out.size[2] - 6) % 3 != 0)
        throw std::runtime_error(
            "unexpected pose output; expected [1,N,6+3*K] float tensor");
    if (!out.isContinuous())
        out = out.clone();

    const int rows = out.size[out.dims - 2];
    const int cols = out.size[out.dims - 1];   // 57
    const int nkpt = (cols - 6) / 3;           // 17
    const float* p = reinterpret_cast<const float*>(out.data);

    std::vector<PoseDetection> res;
    for (int i = 0; i < rows; ++i) {
        const float* row = p + static_cast<size_t>(i) * cols;
        if (row[4] < conf_thres) continue;
        PoseDetection d;
        d.box = cv::Rect2f((row[0] - dw) / r, (row[1] - dh) / r,
                           (row[2] - row[0]) / r, (row[3] - row[1]) / r);
        d.score = row[4];
        d.kpts.resize(nkpt);
        for (int k = 0; k < nkpt; ++k) {
            const float kx = row[6 + k * 3 + 0];
            const float ky = row[6 + k * 3 + 1];
            d.kpts[k] = Keypoint{(kx - dw) / r, (ky - dh) / r, row[6 + k * 3 + 2]};
        }
        res.push_back(std::move(d));
    }
    return res;
}

void draw_pose(cv::Mat& img, const std::vector<PoseDetection>& poses,
               bool skeleton_only, float kpt_thres) {
    int box_th, font_th; double font;
    visual_scale(img, box_th, font, font_th);
    const int limb_th = std::max(2, box_th);
    const int rad = std::max(2, box_th + 1);

    for (size_t pi = 0; pi < poses.size(); ++pi) {
        const auto& d = poses[pi];
        const cv::Scalar col = color_for((int)pi);
        if (!skeleton_only)
            cv::rectangle(img, d.box.tl(), d.box.br(), col, std::max(1, box_th - 1), cv::LINE_AA);
        // limbs
        for (auto& e : SKELETON) {
            const auto& a = d.kpts[e[0]];
            const auto& b = d.kpts[e[1]];
            if (a.conf < kpt_thres || b.conf < kpt_thres) continue;
            cv::line(img, {cvRound(a.x), cvRound(a.y)}, {cvRound(b.x), cvRound(b.y)},
                     col, limb_th, cv::LINE_AA);
        }
        // joints
        for (const auto& k : d.kpts) {
            if (k.conf < kpt_thres) continue;
            cv::circle(img, {cvRound(k.x), cvRound(k.y)}, rad, cv::Scalar(255, 255, 255),
                       cv::FILLED, cv::LINE_AA);
            cv::circle(img, {cvRound(k.x), cvRound(k.y)}, rad, col, std::max(1, box_th - 1),
                       cv::LINE_AA);
        }
        if (!skeleton_only) {
            const std::string name = display_name(CLS_PERSON);
            char lbl[96]; std::snprintf(lbl, sizeof(lbl), "%s %.2f", name.c_str(), d.score);
            draw_label(img, d.box.tl(), lbl, col, font, font_th);
        }
    }
}

// ---------------------------------------------------------------------------
// Instance segmentation
// ---------------------------------------------------------------------------
std::vector<SegDetection> detect_seg(cv::dnn::Net& net, const cv::Mat& img,
                                     int size, float conf_thres) {
    const int H = img.rows, W = img.cols;
    float r; int dw, dh;
    cv::Mat padded = letterbox(img, size, r, dw, dh);
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(size, size),
                                          cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    // Identify which output is [1,300,C] (dets) and which is [1,32,mh,mw] (protos).
    cv::Mat dets, proto;
    for (auto& o : outs) {
        if (o.dims == 3) dets = o;
        else if (o.dims == 4) proto = o;
    }
    if (dets.empty() || proto.empty()) return {};
    if (dets.type() != CV_32F || proto.type() != CV_32F ||
        dets.size[2] < 7 || proto.size[1] <= 0 ||
        dets.size[2] < 6 + proto.size[1])
        throw std::runtime_error(
            "unexpected segmentation outputs; expected [1,N,6+M] and [1,M,H,W]");
    if (!dets.isContinuous())
        dets = dets.clone();
    if (!proto.isContinuous())
        proto = proto.clone();

    const int rows = dets.size[1];
    const int cols = dets.size[2];       // 38
    const int nm   = proto.size[1];      // 32 mask coeffs
    const int mh = proto.size[2], mw = proto.size[3];
    const float* dp = reinterpret_cast<const float*>(dets.data);

    // protos as (nm, mh*mw)
    cv::Mat protoMat(nm, mh * mw, CV_32F, proto.data);

    // letterbox geometry in proto space
    const int nw = (int)std::round(W * r), nh = (int)std::round(H * r);

    std::vector<SegDetection> res;
    for (int i = 0; i < rows; ++i) {
        const float* row = dp + static_cast<size_t>(i) * cols;
        if (row[4] < conf_thres) continue;

        SegDetection d;
        d.box = cv::Rect2f((row[0] - dw) / r, (row[1] - dh) / r,
                           (row[2] - row[0]) / r, (row[3] - row[1]) / r);
        d.score = row[4];
        d.cls   = (int)row[5];

        cv::Mat coeff(1, nm, CV_32F);
        for (int c = 0; c < nm; ++c) coeff.at<float>(0, c) = row[6 + c];
        cv::Mat m = coeff * protoMat;                 // 1 x (mh*mw)
        m = m.reshape(1, mh);                          // mh x mw (proto space)
        cv::Mat prob;
        cv::exp(-m, prob);
        prob = 1.0 / (1.0 + prob);                     // sigmoid

        // proto space -> letterboxed 640 -> remove pad -> original size
        cv::Mat up;
        cv::resize(prob, up, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
        int px = (int)std::round((float)dw), py = (int)std::round((float)dh);
        cv::Rect keep(px, py, nw, nh);
        keep &= cv::Rect(0, 0, size, size);
        cv::Mat cropped = up(keep);
        cv::Mat full;
        cv::resize(cropped, full, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

        cv::Mat bin = full > 0.5f;                     // 8-bit 0/255
        // gate by the detection box so masks don't bleed outside
        cv::Mat gated = cv::Mat::zeros(H, W, CV_8U);
        cv::Rect bb(cvRound(d.box.x), cvRound(d.box.y),
                    cvRound(d.box.width), cvRound(d.box.height));
        bb &= cv::Rect(0, 0, W, H);
        if (bb.width > 0 && bb.height > 0) bin(bb).copyTo(gated(bb));
        d.mask = gated;
        res.push_back(std::move(d));
    }
    return res;
}

void draw_seg(cv::Mat& img, const std::vector<SegDetection>& segs, float alpha) {
    int box_th, font_th; double font;
    visual_scale(img, box_th, font, font_th);

    cv::Mat overlay = img.clone();
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& d = segs[i];
        const cv::Scalar col = color_for(d.cls == CLS_PERSON ? (int)i : d.cls);
        overlay.setTo(col, d.mask);
    }
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img);

    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& d = segs[i];
        const cv::Scalar col = color_for(d.cls == CLS_PERSON ? (int)i : d.cls);
        // mask contour outline
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(d.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(img, contours, -1, col, std::max(1, box_th - 1), cv::LINE_AA);
        const std::string name = display_name(d.cls);
        char lbl[96]; std::snprintf(lbl, sizeof(lbl), "%s %.2f", name.c_str(), d.score);
        draw_label(img, d.box.tl(), lbl, col, font, font_th);
    }
}

}  // namespace y26
