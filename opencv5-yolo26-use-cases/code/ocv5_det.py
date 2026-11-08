"""
ocv5_det.py
Shared building blocks for the OpenCV 5 use-case demos.

Everything here runs on OpenCV 5 + NumPy only. PyTorch / Ultralytics are used
once, to export the ONNX file, and never at runtime.

Contents
    Detector        YOLO26 ONNX model on OpenCV 5's DNN module (NMS-free output)
    IoUTracker      a small multi-object tracker that gives every object an ID
    CameraAligner   keeps lines and zones pinned to the scene when the camera drifts
    Line / Zone     geometry helpers for line crossing and polygon membership
    draw_*          drawing helpers that scale with the frame size
    VideoJob        a read -> process -> write loop with FPS bookkeeping
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# COCO class names, in the order YOLO26 was trained on
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

ENGINES = {
    "auto": cv2.dnn.ENGINE_AUTO,        # new engine, falls back to classic if a layer is unsupported
    "new": cv2.dnn.ENGINE_NEW,          # the rewritten OpenCV 5 engine (CPU)
    "classic": cv2.dnn.ENGINE_CLASSIC,  # the 4.x engine (needed for the CUDA backend)
}


# --------------------------------------------------------------------------- detector
class Detector:
    """YOLO26 on OpenCV 5 DNN.

    YOLO26 is NMS-free: the ONNX output is a fixed [1, 300, 6] tensor where every
    row is already a final detection (x1, y1, x2, y2, score, class_id) in the
    640x640 letterboxed input space. We only filter by score and undo the
    letterbox, no transpose and no cv2.dnn.NMSBoxes.
    """

    def __init__(self, onnx_path, imgsz=640, engine="new", conf=0.30, classes=None):
        self.net = cv2.dnn.readNet(onnx_path, "", "", ENGINES[engine])
        self.imgsz = imgsz
        self.conf = conf
        # keep only these class names (None = keep all 80)
        self.keep = None if classes is None else {COCO.index(c) for c in classes}

    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = round(h * s), round(w * s)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, s, left, top

    def __call__(self, img, conf=None):
        """Return an (N, 6) float array: x1, y1, x2, y2, score, class_id in image pixels."""
        conf = self.conf if conf is None else conf
        canvas, s, left, top = self._letterbox(img)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0]                      # (300, 6)
        out = out[out[:, 4] >= conf]
        if self.keep is not None:
            out = out[np.isin(out[:, 5].astype(int), list(self.keep))]
        if len(out) == 0:
            return np.zeros((0, 6), np.float32)
        out = out.copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - left) / s     # undo letterbox
        out[:, [1, 3]] = (out[:, [1, 3]] - top) / s
        h, w = img.shape[:2]
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w - 1)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h - 1)
        return out

    def tiled(self, img, tiles=(2, 2), overlap=0.2, conf=None, iou=0.5):
        """Detect on overlapping tiles and merge, for small objects in large frames.

        Each tile is run through the network at full model resolution, so a car
        that is 15 px wide in the whole frame becomes ~30 px wide in its tile.
        Duplicates from overlapping tiles are merged with class-aware NMS
        (the only place we need NMS, because tiles see the same object twice).
        """
        h, w = img.shape[:2]
        ty, tx = tiles
        th, tw = int(h / (ty - (ty - 1) * overlap)), int(w / (tx - (tx - 1) * overlap))
        ys = np.linspace(0, h - th, ty).astype(int)
        xs = np.linspace(0, w - tw, tx).astype(int)
        dets = []
        for y0 in ys:
            for x0 in xs:
                d = self(img[y0:y0 + th, x0:x0 + tw], conf)
                d[:, [0, 2]] += x0
                d[:, [1, 3]] += y0
                dets.append(d)
        dets.append(self(img, conf))                    # the full frame catches large objects
        d = np.concatenate(dets)
        if len(d) == 0:
            return d
        boxes = [[float(a), float(b), float(c - a), float(e - b)] for a, b, c, e in d[:, :4]]
        keep = cv2.dnn.NMSBoxesBatched(boxes, d[:, 4].tolist(), d[:, 5].astype(int).tolist(), conf or self.conf, iou)
        return d[np.array(keep, dtype=int).reshape(-1)]


# --------------------------------------------------------------------------- tracker
def iou_matrix(a, b):
    """Pairwise IoU between boxes a (N,4) and b (M,4) in x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


@dataclass
class Track:
    tid: int
    box: np.ndarray                 # x1, y1, x2, y2
    cls: int
    score: float
    hits: int = 1
    missed: int = 0
    trail: list = field(default_factory=list)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))

    @property
    def center(self):
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def foot(self):
        """Bottom-center point: where a person or vehicle touches the ground."""
        return np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])


class IoUTracker:
    """Greedy IoU tracker with constant-velocity prediction.

    Every frame we (1) predict where each track should be, (2) match predictions
    to new detections by IoU, highest first, (3) update matched tracks, start new
    tracks for unmatched detections, and drop tracks that have been missing for
    `max_missed` frames. It is tiny, has no dependencies, and is good enough for
    a fixed camera at normal frame rates.
    """

    def __init__(self, iou_thr=0.3, max_missed=15, min_hits=3, trail_len=40, dist_thr=0.8):
        self.iou_thr, self.max_missed, self.min_hits, self.trail_len = iou_thr, max_missed, min_hits, trail_len
        self.dist_thr = dist_thr                     # max center distance in units of object size (pass 2)
        self.tracks, self.next_id = [], 1

    def _assign(self, t, det):
        new = det[:4]
        steps = t.missed + 1                           # frames since the last real observation
        t.vel = 0.6 * t.vel + 0.4 * (new - (t.box - t.vel * t.missed)) / steps   # smoothed per-frame velocity
        t.box, t.score = new.copy(), float(det[4])
        t.hits, t.missed = t.hits + 1, 0

    def update(self, dets):
        preds = np.array([t.box + t.vel for t in self.tracks]).reshape(-1, 4)
        ious = iou_matrix(preds, dets[:, :4])
        matched_t, matched_d = set(), set()
        # pass 1: greedy matching, best IoU first, same class only
        for ti, di in zip(*np.unravel_index(np.argsort(-ious, axis=None), ious.shape)):
            if ious[ti, di] < self.iou_thr:
                break
            if ti in matched_t or di in matched_d or self.tracks[ti].cls != int(dets[di, 5]):
                continue
            self._assign(self.tracks[ti], dets[di])
            matched_t.add(ti); matched_d.add(di)
        # pass 2: small or fast objects often have zero IoU with their prediction,
        # so match what is left by center distance, relative to the object's size
        rest_t = [i for i in range(len(self.tracks)) if i not in matched_t]
        rest_d = [i for i in range(len(dets)) if i not in matched_d]
        if rest_t and rest_d:
            pc = (preds[rest_t, :2] + preds[rest_t, 2:]) / 2
            dc = (dets[rest_d, :2] + dets[rest_d, 2:4]) / 2
            size = np.maximum(preds[rest_t, 2] - preds[rest_t, 0], preds[rest_t, 3] - preds[rest_t, 1])
            dist = np.linalg.norm(pc[:, None] - dc[None], axis=2) / (size[:, None] + 1e-6)
            for a, b in zip(*np.unravel_index(np.argsort(dist, axis=None), dist.shape)):
                if dist[a, b] > self.dist_thr:
                    break
                ti, di = rest_t[a], rest_d[b]
                if ti in matched_t or di in matched_d or self.tracks[ti].cls != int(dets[di, 5]):
                    continue
                self._assign(self.tracks[ti], dets[di])
                matched_t.add(ti); matched_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.missed += 1
                t.box = t.box + t.vel                   # coast on the prediction
        for di in range(len(dets)):
            if di not in matched_d:
                self.tracks.append(Track(self.next_id, dets[di, :4].copy(), int(dets[di, 5]), float(dets[di, 4])))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for t in self.tracks:
            if t.missed == 0:
                t.trail.append(tuple(t.center.astype(int)))
                t.trail = t.trail[-self.trail_len:]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.missed == 0]


# --------------------------------------------------------------------------- camera drift
class CameraAligner:
    """Keeps a transform from the current frame back to a reference frame.

    Zones and lines are drawn once on the reference frame. If the camera drifts
    (a drone, a pole swaying in the wind, a slow zoom), we match ORB features
    between the current frame and a keyframe, fit a transform with RANSAC, and
    chain it to the keyframe's own transform.

    Two details make this robust:
      * detected objects are masked out before extracting features, so only the
        static background votes (moving cars would otherwise drag the fit along)
      * `motion` picks the model: "homography" (8 DOF) for a flat scene such as a
        parking lot, "similarity" (4 DOF: shift, rotation, zoom) for scenes with
        depth such as a highway, where a full homography over-fits.
    Features are matched on a half-size copy; the result is rescaled to full
    resolution with H_full = S^-1 @ H_small @ S.
    """

    def __init__(self, ref_frame, motion="homography", n_features=1500, rekey_every=20, min_inliers=30,
                 scale=0.5, boxes=None):
        self.motion, self.scale = motion, scale
        self.S = np.diag([scale, scale, 1.0])
        self.Si = np.linalg.inv(self.S)
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.rekey_every, self.min_inliers = rekey_every, min_inliers
        self._set_key(ref_frame, np.eye(3), boxes)
        self.H = np.eye(3)                       # current frame -> reference
        self.inliers, self.n = 0, 0

    def _features(self, frame, boxes):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        mask = np.full(g.shape, 255, np.uint8)
        if boxes is not None:
            for x1, y1, x2, y2 in (np.asarray(boxes)[:, :4] * self.scale).astype(int):
                mask[max(y1, 0):y2, max(x1, 0):x2] = 0          # ignore pixels on detected objects
        return self.orb.detectAndCompute(g, mask)

    def _set_key(self, frame, H_key_to_ref, boxes=None):
        self.kp, self.des = self._features(frame, boxes)
        self.H_key = H_key_to_ref

    def update(self, frame, boxes=None):
        self.n += 1
        kp, des = self._features(frame, boxes)
        if des is not None and self.des is not None and len(des) > 8 and len(self.des) > 8:
            pairs = self.matcher.knnMatch(des, self.des, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
            if len(good) >= 8:
                src = np.float32([kp[m.queryIdx].pt for m in good])
                dst = np.float32([self.kp[m.trainIdx].pt for m in good])
                if self.motion == "similarity":
                    A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    Hs = None if A is None else np.vstack([A, [0, 0, 1]])
                else:
                    Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                self.inliers = int(mask.sum()) if mask is not None else 0
                if Hs is not None and self.inliers >= self.min_inliers:
                    self.H = self.H_key @ (self.Si @ Hs @ self.S)
        if self.n % self.rekey_every == 0:
            self._set_key(frame, self.H, boxes)   # new keyframe keeps the overlap large
        return self.H


def to_ref(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


# --------------------------------------------------------------------------- geometry
def side_of_line(p, a, b):
    """Sign of the cross product: >0 on one side of line a->b, <0 on the other."""
    return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


class LineCounter:
    """Counts tracks whose anchor point crosses segment a->b, in both directions."""

    def __init__(self, a, b, name="line", anchor="center", classes=None):
        self.a, self.b, self.name, self.anchor = np.array(a, float), np.array(b, float), name, anchor
        self.classes = None if classes is None else {COCO.index(c) for c in classes}
        self.last_side, self.counted = {}, {}
        self.count_in, self.count_out = 0, 0
        self.by_class = {}
        self.flash = 0

    def _within_segment(self, p):
        ab = self.b - self.a
        t = np.dot(p - self.a, ab) / (np.dot(ab, ab) + 1e-9)
        return -0.05 <= t <= 1.05

    def update(self, tracks, H=None):
        """H (optional) maps current-frame points into the frame the line was drawn on."""
        for t in tracks:
            if self.classes is not None and t.cls not in self.classes:
                continue
            p = t.foot if self.anchor == "foot" else t.center
            if H is not None:
                p = to_ref(H, [p])[0]
            s = side_of_line(p, self.a, self.b)
            prev, streak = self.last_side.get(t.tid, (None, 0))
            self.last_side[t.tid] = (s, streak + 1 if s == prev else 1)
            # a crossing must come from a side the object stayed on for at least 2 updates,
            # which filters out box jitter right on the line and tracks that start on top of it
            if prev is None or s == 0 or prev == 0 or s == prev or streak < 2 or t.tid in self.counted:
                continue
            if not self._within_segment(p):
                continue
            direction = "in" if s > 0 else "out"
            self.counted[t.tid] = direction
            if direction == "in":
                self.count_in += 1
            else:
                self.count_out += 1
            name = COCO[t.cls]
            self.by_class.setdefault(name, [0, 0])[0 if direction == "in" else 1] += 1
            self.flash = 6


class Zone:
    """A named polygon. Membership is tested with cv2.pointPolygonTest."""

    def __init__(self, pts, name="zone", color=(0, 200, 255)):
        self.pts = np.array(pts, np.int32)
        self.name, self.color = name, color

    def contains(self, p):
        return cv2.pointPolygonTest(self.pts, (float(p[0]), float(p[1])), False) >= 0


def scaled(pts, w, h):
    """Polygons are defined in normalized 0..1 coordinates so they work at any resolution."""
    return [(int(x * w), int(y * h)) for x, y in pts]


# --------------------------------------------------------------------------- drawing
PALETTE = [(255, 99, 71), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]


def ui_scale(img):
    """A size factor so lines and text stay readable from 480p to 4K."""
    return max(img.shape[0] / 720.0, 0.6)


def color_for(i):
    return PALETTE[i % len(PALETTE)]


def draw_label(img, text, org, color, scale=1.0, text_color=(255, 255, 255)):
    fs, th = 0.55 * scale, max(1, int(round(1.5 * scale)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x, y = int(org[0]), int(org[1])
    y = max(y, tht + base + 4)
    x = max(0, min(x, img.shape[1] - tw - 6))          # keep the label inside the frame
    cv2.rectangle(img, (x, y - tht - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, th, cv2.LINE_AA)


def draw_box(img, box, color, label=None, scale=1.0, thickness=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness or max(2, int(2 * scale)), cv2.LINE_AA)
    if label:
        draw_label(img, label, (x1, y1), color, scale)


def draw_trail(img, trail, color, scale=1.0):
    for i in range(1, len(trail)):
        t = max(1, int(scale * 3 * i / len(trail)))
        cv2.line(img, trail[i - 1], trail[i], color, t, cv2.LINE_AA)


def draw_zone(img, zone, alpha=0.25, color=None, label=True, scale=1.0, label_pos="top"):
    """label_pos "top" puts the name at the polygon's top-left; "bottom" at its lowest edge, which
    reads better for lane quads that taper towards the horizon."""
    color = color or zone.color
    overlay = img.copy()
    cv2.fillPoly(overlay, [zone.pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [zone.pts], True, color, max(2, int(2 * scale)), cv2.LINE_AA)
    if label:
        if label_pos == "bottom":
            ymax = zone.pts[:, 1].max()
            low = zone.pts[zone.pts[:, 1] >= ymax - 0.05 * img.shape[0]]
            x, y = int(low[:, 0].mean()) - int(40 * scale), min(int(ymax) - int(8 * scale), img.shape[0] - int(8 * scale))
        else:
            x, y = zone.pts[:, 0].min(), zone.pts[:, 1].min()
        draw_label(img, zone.name, (max(x, 0), y), color, scale)


def draw_panel(img, lines, org=(12, 12), scale=1.0, bg=(20, 20, 20), alpha=0.65, width=None):
    """A translucent info panel with one text line per entry. Entries can be (text, color)."""
    fs, th = 0.62 * scale, max(1, int(round(1.6 * scale)))
    lh = int(30 * scale)
    items = [(l, (255, 255, 255)) if isinstance(l, str) else l for l in lines]
    w = width or max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for t, _ in items) + int(24 * scale)
    h = lh * len(items) + int(14 * scale)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    for i, (t, c) in enumerate(items):
        cv2.putText(img, t, (x + int(12 * scale), y + int(8 * scale) + lh * (i + 1) - int(8 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
    return w, h


# --------------------------------------------------------------------------- video loop
class VideoJob:
    """Read a video, call process(frame, idx) -> frame, and write the result.

    Keeps a running FPS for the whole pipeline (decode + detect + logic + draw)
    and can stop early with max_seconds so demos stay quick on CPU.
    """

    def __init__(self, src, dst, max_seconds=None, out_width=None, every=1):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {src}")
        self.fps_in = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.every = every                              # process every n-th frame (for 50/60 fps clips)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_w = out_width or self.w
        self.out_h = int(round(self.h * self.out_w / self.w / 2) * 2)
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_frames = n if not max_seconds else min(n if n > 0 else 10**9, int(max_seconds * self.fps_in))
        self.writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), self.fps_in / every, (self.out_w, self.out_h))
        self.dst = dst
        self.proc_times = []

    def run(self, process, log_every=100):
        idx, t_start = 0, time.perf_counter()
        while idx < self.max_frames:
            ok, frame = self.cap.read()
            if not ok:
                break
            if idx % self.every == 0:
                if frame.shape[1] != self.out_w:
                    frame = cv2.resize(frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
                t0 = time.perf_counter()
                frame = process(frame, idx)
                self.proc_times.append(time.perf_counter() - t0)
                self.writer.write(frame)
            idx += 1
            if log_every and idx % log_every == 0:
                print(f"  frame {idx}/{self.max_frames}  {self.fps:.1f} FPS")
        self.cap.release()
        self.writer.release()
        print(f"done: {len(self.proc_times)} frames, {self.fps:.1f} FPS average, total {time.perf_counter() - t_start:.1f}s -> {self.dst}")
        return self

    @property
    def fps(self):
        return len(self.proc_times) / max(sum(self.proc_times), 1e-6)
