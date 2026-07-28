// detect_teams.cpp
//
// Football-specific post-processing on top of YOLO26: split the detected
// players into two teams by the colour of their shirts.
//
// Pipeline per frame:
//   1. YOLO26 -> person + sports-ball detections (OpenCV 5 DNN, NMS-free).
//   2. For every "person", sample a torso patch (upper-centre of the box) and
//      reduce it to a single Lab colour, ignoring green pitch pixels.
//   3. k-means (k = --teams, default 2) over those torso colours -> team id.
//   4. Draw each player in their team colour; the ball stays yellow.
//
// The detector never knew what a "team" was; all of the football knowledge
// lives in a few lines of classic OpenCV on top of the boxes. Works on a single
// image or a video (auto-detected).
//
// Usage (run from the repo root; no args needed for a default demo run):
//   detect_teams --model models/yolo26n_640.onnx --imgsz 640
//                --source assets/videos/stadium_wide.mp4 --teams 2
//                --out outputs/team_split.mp4
#include "yolo26_dnn.hpp"
#include "cli.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <vector>

// Team palette (BGR): 0=red, 1=cyan/blue, 2=green, 3=magenta.
static const cv::Scalar TEAM_COLORS[4] = {
    {60, 60, 220}, {230, 180, 40}, {60, 200, 80}, {200, 60, 200}};

// Reduce a player's torso to one Lab colour. Samples the upper-centre of the
// box (the shirt), drops obviously-green pitch pixels, returns mean Lab.
static bool torso_color(const cv::Mat& bgr, const cv::Rect2f& box, cv::Vec3f& lab_out) {
    cv::Rect roi(cvRound(box.x + 0.25f * box.width),
                 cvRound(box.y + 0.15f * box.height),
                 cvRound(0.50f * box.width),
                 cvRound(0.35f * box.height));
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.width < 3 || roi.height < 3) return false;

    cv::Mat patch = bgr(roi), hsv, lab;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(patch, lab, cv::COLOR_BGR2Lab);

    // Mask out grass: hue ~ 35..85 with decent saturation.
    cv::Mat green;
    cv::inRange(hsv, cv::Scalar(35, 40, 40), cv::Scalar(85, 255, 255), green);
    cv::Mat keep = ~green;

    cv::Scalar m = cv::mean(lab, keep);
    if (cv::countNonZero(keep) < 10) m = cv::mean(lab);  // fall back to whole patch
    lab_out = cv::Vec3f((float)m[0], (float)m[1], (float)m[2]);
    return true;
}

// Assign a team id to every player via k-means on their torso Lab colours.
// Cluster ids are re-ordered by the 'a' (green<->red) axis so team 0 stays the
// same team across frames instead of flickering with k-means' arbitrary labels.
static std::vector<int> assign_teams(const std::vector<cv::Vec3f>& colors, int k) {
    const int n = (int)colors.size();
    std::vector<int> team(n, 0);
    if (n < k || k < 2) return team;

    cv::Mat samples(n, 3, CV_32F);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j) samples.at<float>(i, j) = colors[i][j];

    cv::Mat labels, centers;
    cv::kmeans(samples, k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Stable ordering: sort clusters by center 'a' channel (col 1).
    std::vector<int> order(k);
    for (int i = 0; i < k; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return centers.at<float>(a, 1) < centers.at<float>(b, 1); });
    std::vector<int> remap(k);
    for (int rank = 0; rank < k; ++rank) remap[order[rank]] = rank;

    for (int i = 0; i < n; ++i) team[i] = remap[labels.at<int>(i)];
    return team;
}

static void process(cv::dnn::Net& net, cv::Mat& frame, int imgsz, float conf, int k, float minh) {
    auto dets = y26::detect(net, frame, imgsz, conf);

    // Drop tiny/distant person boxes (spectators in the stands) so the jersey
    // clustering focuses on the players on the pitch.
    const float min_px = minh * frame.rows;

    std::vector<int> person_idx;
    std::vector<cv::Vec3f> colors;
    for (int i = 0; i < (int)dets.size(); ++i) {
        if (dets[i].cls != y26::CLS_PERSON) continue;
        if (dets[i].box.height < min_px) continue;
        cv::Vec3f lab;
        if (torso_color(frame, dets[i].box, lab)) {
            person_idx.push_back(i);
            colors.push_back(lab);
        }
    }
    std::vector<int> team = assign_teams(colors, k);

    int box_th, font_th; double font;
    y26::visual_scale(frame, box_th, font, font_th);

    auto labelled = [&](const cv::Rect2f& box, const cv::Scalar& col, const std::string& text) {
        cv::rectangle(frame, box.tl(), box.br(), col, box_th, cv::LINE_AA);
        int bl = 0; cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font, font_th, &bl);
        cv::Point tl(cvRound(box.x), cvRound(box.y));
        cv::rectangle(frame, {tl.x, tl.y - ts.height - bl - 4}, {tl.x + ts.width + 2, tl.y}, col, cv::FILLED);
        cv::putText(frame, text, {tl.x + 1, tl.y - bl - 2}, cv::FONT_HERSHEY_SIMPLEX, font,
                    {0, 0, 0}, font_th, cv::LINE_AA);
    };

    // Players, coloured by team.
    for (int p = 0; p < (int)person_idx.size(); ++p) {
        const auto& d = dets[person_idx[p]];
        std::string lbl = std::string("Team ") + char('A' + team[p]);
        labelled(d.box, TEAM_COLORS[team[p] % 4], lbl);
    }
    // The ball, drawn distinctly.
    for (const auto& d : dets) {
        if (d.cls != y26::CLS_SPORTS_BALL) continue;
        labelled(d.box, cv::Scalar(0, 255, 255), "ball");
    }
}

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    const int imgsz          = cli.geti("imgsz", 640);
    const std::string model  = cli.get("model", "models/yolo26n_640.onnx");
    const std::string source = cli.get("source", "assets/videos/stadium_wide.mp4");
    const std::string engine = cli.get("engine", "auto");
    const float conf         = cli.getf("conf", 0.30f);
    const int teams          = cli.geti("teams", 2);
    const float minh         = cli.getf("minh", 0.12f);
    const std::string out    = cli.get("out", "outputs/team_split.mp4");
    if (teams < 2 || teams > 4) {
        std::cerr << "--teams must be between 2 and 4\n";
        return 1;
    }

    cv::dnn::Net net = y26::build_net(model, engine);
    y26::ensure_parent_dir(out);

    // Image mode if imread succeeds, else treat the source as a video.
    cv::Mat img = cv::imread(source);
    if (!img.empty()) {
        process(net, img, imgsz, conf, teams, minh);
        if (!cv::imwrite(out, img)) {
            std::cerr << "could not write output: " << out << "\n";
            return 1;
        }
        std::cout << "saved " << out << "\n";
        return 0;
    }

    cv::VideoCapture cap(source);
    if (!cap.isOpened()) { std::cerr << "cannot open source: " << source << "\n"; return 1; }
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, {W, H});
    if (!writer.isOpened()) {
        std::cerr << "could not open writer: " << out << "\n";
        return 1;
    }

    cv::Mat frame; int n = 0;
    while (cap.read(frame)) { process(net, frame, imgsz, conf, teams, minh); writer.write(frame); ++n; }
    if (n == 0) {
        std::cerr << "video contained no readable frames: " << source << "\n";
        return 1;
    }
    std::cout << "processed " << n << " frames, saved " << out << "\n";
    return 0;
}
