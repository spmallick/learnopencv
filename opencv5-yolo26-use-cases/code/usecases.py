"""
usecases.py
Three real-world applications built on the same OpenCV 5 + YOLO26 detector.

    count_traffic     line-crossing counts per class and direction
    monitor_parking   parked vs moving cars per parking area, with camera-drift compensation
    zone_alert        restricted-zone alerts with dwell time, snapshots, and an event log

Every function takes a plain config dict, so the same code runs every example
in the notebook. Coordinates in configs are normalized (0..1) so they survive
any resize.
"""

import csv
import os
from collections import deque

import cv2
import numpy as np

from ocv5_det import (COCO, CameraAligner, Detector, to_ref, IoUTracker, LineCounter, VideoJob, Zone, color_for, draw_box,
                      draw_label, draw_panel, draw_trail, draw_zone, scaled, ui_scale)

GREEN, RED, AMBER, WHITE, CYAN = (80, 200, 60), (40, 40, 230), (0, 190, 255), (255, 255, 255), (230, 200, 40)
CLASS_COLORS = {"car": (255, 140, 40), "truck": (40, 170, 255), "bus": (200, 80, 220),
                "motorcycle": (60, 220, 120), "bicycle": (200, 220, 60), "person": (90, 90, 250)}


def class_color(cls_id):
    return CLASS_COLORS.get(COCO[cls_id], color_for(cls_id))


# =========================================================================== 1. traffic counting
def count_traffic(cfg, detector=None):
    """Count objects crossing one or more lines.

    cfg keys
        video, out             input and output paths
        model, imgsz, conf     detector settings (ignored if `detector` is passed)
        classes                list of COCO names to keep, e.g. ["car", "truck", "bus"]
        lines                  list of dicts {name, a, b, labels=("in","out"), anchor="center"|"foot"}
        max_seconds, every     clip length and frame stride
        align                  True if the camera drifts: lines stay pinned to the road (ORB + homography)
        merge_riders           True to fold a rider's "person" box into the motorcycle/bicycle under it
        csv                    optional path for a per-crossing event log
    """
    det = detector or Detector(cfg["model"], cfg.get("imgsz", 640), conf=cfg.get("conf", 0.3), classes=cfg["classes"])
    tracker = IoUTracker(iou_thr=cfg.get("iou", 0.25), max_missed=cfg.get("max_missed", 12), min_hits=3)
    job = VideoJob(cfg["video"], cfg["out"], cfg.get("max_seconds"), cfg.get("out_width"), cfg.get("every", 1))
    W, H = job.out_w, job.out_h
    counters = [LineCounter(scaled([l["a"]], W, H)[0], scaled([l["b"]], W, H)[0], l["name"], l.get("anchor", "center"),
                            l.get("classes")) for l in cfg["lines"]]
    labels = [l.get("labels", ("in", "out")) for l in cfg["lines"]]
    events = []
    s = ui_scale(np.zeros((H, W), np.uint8))
    state = {"aligner": None, "riders": 0}

    def process(frame, idx):
        dets = det(frame)
        Hc = None
        if cfg.get("align"):
            if state["aligner"] is None:
                state["aligner"] = CameraAligner(frame, motion=cfg.get("align_motion", "similarity"), boxes=dets)
            Hc = state["aligner"].update(frame, dets)
        if cfg.get("merge_riders"):
            dets = merge_riders(dets)
        tracks = tracker.update(dets)
        before = [(c.count_in, c.count_out) for c in counters]
        for c in counters:
            c.update(tracks, Hc)
        # log new crossings
        for c, (bi, bo), lab in zip(counters, before, labels):
            if (c.count_in, c.count_out) != (bi, bo):
                for tid, d in list(c.counted.items())[-((c.count_in - bi) + (c.count_out - bo)):]:
                    t = next((t for t in tracker.tracks if t.tid == tid), None)
                    events.append({"time_s": round(idx / job.fps_in, 2), "line": c.name, "track_id": tid,
                                   "class": COCO[t.cls] if t else "?", "direction": lab[0] if d == "in" else lab[1]})
        # draw
        for t in tracks:
            col = class_color(t.cls)
            counted = any(t.tid in c.counted for c in counters)
            draw_trail(frame, t.trail, col, s)
            draw_box(frame, t.box, col, f"{COCO[t.cls]} #{t.tid}", s * 0.85, thickness=max(2, int(s * (3 if counted else 2))))
        for c, lab in zip(counters, labels):
            col = AMBER if c.flash > 0 else WHITE
            c.flash = max(0, c.flash - 1)
            ends = np.array([c.a, c.b]) if Hc is None else to_ref(np.linalg.inv(Hc), [c.a, c.b])
            a, b = tuple(ends[0].astype(int)), tuple(ends[1].astype(int))
            cv2.line(frame, a, b, (0, 0, 0), max(4, int(6 * s)), cv2.LINE_AA)
            cv2.line(frame, a, b, col, max(2, int(3 * s)), cv2.LINE_AA)
            draw_label(frame, f"{c.name}: {lab[0]} {c.count_in} | {lab[1]} {c.count_out}",
                       (min(a[0], b[0]), min(a[1], b[1]) - 6), (40, 40, 40), s * 0.9)
        # totals per class across all lines
        totals = {}
        for c in counters:
            for k, (i, o) in c.by_class.items():
                totals[k] = totals.get(k, 0) + i + o
        rows = [("TRAFFIC COUNT", AMBER)]
        rows += [(f"{k:<11s}{v:>4d}", class_color(COCO.index(k))) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])]
        rows += [f"total      {sum(totals.values()):>4d}", (f"on screen  {len(tracks):>4d}", (200, 200, 200)),
                 (f"{job.fps:5.1f} FPS  OpenCV 5", (160, 160, 160))]
        draw_panel(frame, rows, (int(12 * s), int(12 * s)), s * 0.9)
        return frame

    job.run(process)
    if cfg.get("csv"):
        _write_csv(cfg["csv"], events, ["time_s", "line", "track_id", "class", "direction"])
    summary = {c.name: {"in": c.count_in, "out": c.count_out, "by_class": c.by_class} for c in counters}
    return {"summary": summary, "events": events, "fps": job.fps, "frames": len(job.proc_times)}


def merge_riders(dets):
    """Drop "person" boxes that are riders sitting on a detected motorcycle or bicycle.

    COCO labels the rider and the bike separately. On a busy street that doubles
    the clutter and, worse, a rider can be counted as a pedestrian. A person is a
    rider when they overlap a two-wheeler horizontally and their feet end inside
    the bike's box.
    """
    bikes = dets[np.isin(dets[:, 5], [COCO.index("motorcycle"), COCO.index("bicycle")])]
    if len(bikes) == 0:
        return dets
    keep = []
    for d in dets:
        if int(d[5]) != COCO.index("person"):
            keep.append(True)
            continue
        pw = d[2] - d[0]
        ov = np.clip(np.minimum(d[2], bikes[:, 2]) - np.maximum(d[0], bikes[:, 0]), 0, None) / max(pw, 1)
        feet_in = (d[3] > bikes[:, 1]) & (d[3] < bikes[:, 3] + 0.15 * (bikes[:, 3] - bikes[:, 1]))
        keep.append(not np.any((ov > 0.5) & feet_in))
    return dets[np.array(keep)]


# =========================================================================== 2. parking
def monitor_parking(cfg, detector=None):
    """Count parked cars per parking area and tell them apart from moving cars.

    cfg keys
        video, out, model, imgsz, conf, max_seconds, every
        areas          list of {name, pts (normalized, on the first frame), capacity (optional)}
        align          True to compensate camera drift with ORB + homography
        park_speed     max speed (fraction of frame width per second) to call a car parked
        chart          optional path for an occupancy-over-time PNG
    """
    det = detector or Detector(cfg["model"], cfg.get("imgsz", 640), conf=cfg.get("conf", 0.25),
                               classes=cfg.get("classes", ["car", "truck", "bus"]))
    tracker = IoUTracker(iou_thr=0.2, max_missed=20, min_hits=3, trail_len=30)
    job = VideoJob(cfg["video"], cfg["out"], cfg.get("max_seconds"), cfg.get("out_width"), cfg.get("every", 1))
    W, H = job.out_w, job.out_h
    s = ui_scale(np.zeros((H, W), np.uint8))
    areas = [dict(a, zone=Zone(scaled(a["pts"], W, H), a["name"])) for a in cfg["areas"]]
    fps = job.fps_in / job.every
    hist = {}                                       # track id -> deque of reference-frame ground points
    window = max(3, int(fps))                       # one second of history
    speed_thr = cfg.get("park_speed", 0.012) * W     # pixels per second in the reference frame
    timeline = []
    state = {"aligner": None}
    moving_state = {}                               # track id -> currently moving?
    area_hist = {}                                  # area name -> recent raw counts

    def process(frame, idx):
        dets = det(frame)
        if state["aligner"] is None and cfg.get("align", True):
            state["aligner"] = CameraAligner(frame, motion="homography", boxes=dets)   # a parking lot is a plane
        Hc = state["aligner"].update(frame, dets) if state["aligner"] else np.eye(3)
        Hinv = np.linalg.inv(Hc)
        tracker.update(dets)
        # A parked car that misses a detection for a few frames is still there, so occupancy
        # uses every confirmed track, including those coasting through a short dropout.
        tracks = [t for t in tracker.tracks if t.hits >= tracker.min_hits]

        # classify each car as parked or moving using its motion in the reference frame
        parked, moving = [], []
        edge = 0.015 * W
        for t in tracks:
            g = to_ref(Hc, [t.foot - [0, (t.box[3] - t.box[1]) * 0.25]])[0]   # a point on the car's footprint
            q = hist.setdefault(t.tid, deque(maxlen=window))
            # Only real, uncut observations feed the motion history: a coasting track is a guess,
            # and a box clipped by the frame border grows as the car slides into view, which
            # looks like motion when the camera pans.
            cut = t.box[0] < edge or t.box[2] > W - edge or t.box[1] < edge or t.box[3] > H - edge
            if t.missed == 0 and not cut:
                q.append(g)
            # Judge motion only with ~1 s of history, and compare the median position of the
            # first and last quarter of that window. Two-point speeds over a few frames turn
            # a 3 px box jitter into "45 px/s" and flag parked cars as moving.
            if len(q) >= int(0.8 * window):
                pts = np.array(q)
                k = max(2, len(pts) // 4)
                speed = np.linalg.norm(np.median(pts[-k:], 0) - np.median(pts[:k], 0)) * fps / (len(pts) - k)
                was = moving_state.get(t.tid, False)
                moving_state[t.tid] = speed > (0.5 * speed_thr if was else speed_thr)   # hysteresis
            (moving if moving_state.get(t.tid, False) else parked).append((t, g))

        counts = {}
        for a in areas:
            pts_now = to_ref(Hinv, a["zone"].pts).astype(np.int32)   # area drawn where it is in this frame
            if visible_fraction(pts_now, W, H) < 0.5:                # the camera has panned away from it
                counts[a["name"]] = None
                continue
            raw = sum(1 for _, g in parked if a["zone"].contains(g))
            recent = area_hist.setdefault(a["name"], deque(maxlen=max(3, int(fps // 2))))
            recent.append(raw)
            n = int(round(np.median(recent)))                     # half-second median for a steady readout
            counts[a["name"]] = n
            cap = a.get("capacity")
            full = cap is not None and n >= cap
            col = RED if full else (GREEN if cap is None or n < 0.8 * cap else AMBER)
            z = Zone(pts_now, f"{a['name']}: {n}" + (f"/{cap}" if cap else " parked"), col)
            draw_zone(frame, z, alpha=0.18, scale=s * 0.9)
        timeline.append((idx / job.fps_in, dict(counts), len(moving)))

        for t, _ in parked:
            draw_box(frame, t.box, GREEN, None, s * 0.7, thickness=max(1, int((2 if t.missed == 0 else 1) * s)))
        for t, _ in moving:
            draw_trail(frame, t.trail, AMBER, s)
            draw_box(frame, t.box, AMBER, f"moving #{t.tid}", s * 0.8, thickness=max(2, int(3 * s)))

        rows = [("PARKING MONITOR", AMBER)]
        rows += [f"{k:<12s}{v:>4d}" if v is not None else (f"{k:<12s} out of view", (150, 150, 150))
                 for k, v in counts.items()]
        rows += [(f"parked     {sum(v for v in counts.values() if v):>5d}", GREEN), (f"moving     {len(moving):>5d}", AMBER)]
        if state["aligner"]:
            dx, dy = to_ref(Hinv, [[W / 2, H / 2]])[0] - [W / 2, H / 2]
            rows.append((f"drift {dx:+5.0f},{dy:+5.0f} px", (170, 170, 170)))
        rows.append((f"{job.fps:5.1f} FPS  OpenCV 5", (160, 160, 160)))
        draw_panel(frame, rows, (int(12 * s), int(12 * s)), s * 0.9)
        return frame

    job.run(process)
    if cfg.get("chart"):
        _occupancy_chart(timeline, cfg["chart"])
    return {"timeline": timeline, "fps": job.fps, "frames": len(job.proc_times)}


def anchor_point(det, anchor="footprint"):
    """Where an object touches the ground: foot = bottom center; footprint = a little above it."""
    x1, y1, x2, y2 = det[:4]
    return ((x1 + x2) / 2, y2) if anchor == "foot" else ((x1 + x2) / 2, y2 - (y2 - y1) * 0.25)


def count_image(img_path, out_path, areas, detector, tiles=None, merge=False, anchor="footprint",
                title="COUNT (still image)", color=GREEN, label_pos="top"):
    """Count objects per area and per class on a single photo.

    Works for lanes on a highway photo, blocks of a parking lot, or a whole road.
    `tiles` switches on tiled inference for small objects; `merge` folds riders
    into their motorbikes. Returns ({area: {class: n}}, outside, dets).
    """
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    s = ui_scale(img)
    dets = detector.tiled(img, tiles) if tiles else detector(img)
    if merge:
        dets = merge_riders(dets)
    zones = [Zone(scaled(a["pts"], W, H), a["name"]) for a in areas]
    counts, outside = {a["name"]: {} for a in areas}, 0
    for d in dets:
        p = anchor_point(d, anchor)
        z = next((z for z in zones if z.contains(p)), None)
        if z is None:
            outside += 1
            continue
        name = COCO[int(d[5])]
        counts[z.name][name] = counts[z.name].get(name, 0) + 1
    for z in zones:
        n = sum(counts[z.name].values())
        draw_zone(img, Zone(z.pts, f"{z.name}: {n}", color), alpha=0.15, scale=s * 0.9, label_pos=label_pos)
    for d in dets:
        draw_box(img, d[:4], class_color(int(d[5])), None, s * 0.7, thickness=max(1, int(2 * s)))
    rows = [(title, AMBER)]
    for z in zones:
        c = counts[z.name]
        detail = ", ".join(f"{v} {k}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
        rows.append(f"{z.name:<16s}{sum(c.values()):>4d}   {detail}")
    rows.append(f"{'outside areas':<16s}{outside:>4d}")
    mode = f"({tiles[0]}x{tiles[1]} tiles)" if tiles else "(single pass)"
    rows.append((f"{'all objects':<16s}{len(dets):>4d}   {mode}", (200, 200, 200)))
    draw_panel(img, rows, (int(12 * s), int(12 * s)), s * 0.9)
    cv2.imwrite(out_path, img)
    return counts, outside, dets


def parking_image(img_path, out_path, areas, detector, tiles=None):
    """Single-image parking: cars per area on a still photo (a thin wrapper around count_image)."""
    counts, outside, dets = count_image(img_path, out_path, areas, detector, tiles, title="PARKING (still image)")
    return {k: sum(v.values()) for k, v in counts.items()}, len(dets)


def zone_image(img_path, out_path, zones_cfg, detector, tiles=None, title="SAFETY ZONES (still image)"):
    """People standing inside restricted polygons on a single photo. Returns ({zone: n}, people)."""
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    s = ui_scale(img)
    dets = detector.tiled(img, tiles) if tiles else detector(img)
    people = [d for d in dets if COCO[int(d[5])] == "person"]
    zones = [Zone(scaled(z["pts"], W, H), z["name"], RED) for z in zones_cfg]
    inside = {z.name: 0 for z in zones}
    where = []
    for p in people:
        foot = anchor_point(p, "foot")
        z = next((z for z in zones if z.contains(foot)), None)
        where.append((p, foot, z))
        if z is not None:
            inside[z.name] += 1
    for z in zones:
        hot = inside[z.name] > 0
        draw_zone(img, Zone(z.pts, f"{z.name}: {inside[z.name]} inside", RED if hot else AMBER),
                  alpha=0.3 if hot else 0.15, scale=s * 0.9)
    for p, foot, z in where:
        draw_box(img, p[:4], RED if z else GREEN, "ALERT" if z else None, s * 0.8,
                 thickness=max(2, int((3 if z else 2) * s)))
        cv2.circle(img, (int(foot[0]), int(foot[1])), max(3, int(4 * s)), WHITE, -1, cv2.LINE_AA)
    rows = [(title, AMBER)] + [(f"{k:<24s}{v:>3d}", RED if v else GREEN) for k, v in inside.items()]
    rows.append((f"{'people in photo':<24s}{len(people):>3d}", (200, 200, 200)))
    draw_panel(img, rows, (int(12 * s), int(12 * s)), s * 0.9)
    cv2.imwrite(out_path, img)
    return inside, len(people)


def visible_fraction(pts, W, H, step=8):
    """Share of a polygon's area that is inside the frame (coarse raster test)."""
    area = cv2.contourArea(pts.astype(np.float32))
    if area < 1:
        return 0.0
    mask = np.zeros((H // step + 1, W // step + 1), np.uint8)
    cv2.fillPoly(mask, [(pts // step).astype(np.int32)], 1)
    return min(1.0, float(mask.sum()) * step * step / area)


def _rolling_median(x, k=15):
    x = np.array([np.nan if v is None else v for v in x], float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k // 2): i + k // 2 + 1]
        if np.isfinite(x[i]) and np.isfinite(w).any():
            out[i] = np.nanmedian(w)
    return out


def _occupancy_chart(timeline, path):
    import matplotlib.pyplot as plt          # no matplotlib.use("Agg") here: it would switch off inline plots in a notebook
    t = [x[0] for x in timeline]
    names = list(timeline[0][1].keys()) if timeline else []
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=120)
    # a half-second rolling median removes single-frame detection flicker; gaps = area out of view
    for n in names:
        ax.plot(t, _rolling_median([x[1][n] for x in timeline]), label=n, lw=2)
    ax.plot(t, _rolling_median([x[2] for x in timeline]), label="moving", lw=1.5, ls="--", color="tab:orange")
    ax.set_xlabel("time (s)"); ax.set_ylabel("cars"); ax.set_title("Parked cars per area over time")
    ax.grid(alpha=0.3); ax.legend(ncol=len(names) + 1, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =========================================================================== 3. zone alerts
def zone_alert(cfg, detector=None):
    """Raise an alert when a person stays inside a restricted zone.

    Two kinds of zone:
        static   a polygon fixed in the image (platform edge, machine area)
        dynamic  a safety bubble around a moving object, e.g. a forklift (COCO "truck"),
                 rebuilt every frame from that object's box plus a margin

    A person is "in" a zone when their foot point (bottom-center of the box) is
    inside it, which is where they actually stand. An alert fires after
    `dwell_s` seconds inside, saves a snapshot, and is written to the event log.

    cfg keys
        video, out, model, imgsz, conf, max_seconds, every
        zones          list of {name, pts (normalized)}            (static zones)
        bubble         {cls: "truck", margin: 0.6, name}          (optional dynamic zone)
        dwell_s        seconds inside before the alert fires (0 = immediately)
        snapshots      folder for alert snapshots
        csv            path for the event log
    """
    keep = ["person"] + ([cfg["bubble"]["cls"]] if cfg.get("bubble") else [])
    det = detector or Detector(cfg["model"], cfg.get("imgsz", 640), conf=cfg.get("conf", 0.35), classes=keep)
    tracker = IoUTracker(iou_thr=0.25, max_missed=15, min_hits=3)
    job = VideoJob(cfg["video"], cfg["out"], cfg.get("max_seconds"), cfg.get("out_width"), cfg.get("every", 1))
    W, H = job.out_w, job.out_h
    s = ui_scale(np.zeros((H, W), np.uint8))
    zones = [Zone(scaled(z["pts"], W, H), z["name"], RED) for z in cfg.get("zones", [])]
    dwell_need = cfg.get("dwell_s", 0.5)
    inside_since, alerted, events = {}, set(), []
    snap_dir = cfg.get("snapshots")
    if snap_dir:
        os.makedirs(snap_dir, exist_ok=True)
    stats = {"alerts": 0, "max_in_zone": 0}

    def bubble_zones(tracks):
        b = cfg.get("bubble")
        if not b:
            return []
        out = []
        for t in tracks:
            if COCO[t.cls] != b["cls"]:
                continue
            x1, y1, x2, y2 = t.box
            m = b.get("margin", 0.5) * (x2 - x1)
            # Ground-level ellipse around the vehicle's footprint. On an angled camera the floor is
            # foreshortened: a metre towards or away from the camera covers far fewer pixels than a
            # metre sideways, so the bubble is much flatter than it is wide. A tall ellipse would
            # catch people walking in the aisle *behind* the forklift.
            cx, cy = (x1 + x2) / 2, y2 - (y2 - y1) * 0.1
            ax_, ay_ = (x2 - x1) / 2 + m, (y2 - y1) * 0.15 + m * cfg["bubble"].get("depth_ratio", 0.25)
            pts = cv2.ellipse2Poly((int(cx), int(cy)), (int(ax_), int(ay_)), 0, 0, 360, 10)
            out.append((t, Zone(pts, f"{b.get('name', 'safety bubble')} #{t.tid}", AMBER)))
        return out

    def process(frame, idx):
        now = idx / job.fps_in
        tracks = tracker.update(det(frame))
        vehicles = bubble_zones(tracks)
        people = [t for t in tracks if COCO[t.cls] == "person"]
        all_zones = [(None, z) for z in zones] + vehicles

        # people riding the vehicle (box mostly inside the vehicle box) are the operator, not a hazard
        def is_operator(p):
            for v, _ in vehicles:
                ix = max(0, min(p.box[2], v.box[2]) - max(p.box[0], v.box[0]))
                iy = max(0, min(p.box[3], v.box[3]) - max(p.box[1], v.box[1]))
                if ix * iy > 0.6 * (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]):
                    return True
            return False

        hot = set()
        in_zone_now = 0
        for p in people:
            if is_operator(p):
                draw_box(frame, p.box, (150, 150, 150), "operator", s * 0.75)
                continue
            zname = next((z.name for v, z in all_zones if z.contains(p.foot)), None)
            if zname:
                in_zone_now += 1
                key = (p.tid, zname.split(" #")[0])
                inside_since.setdefault(key, now)
                dwell = now - inside_since[key]
                if dwell >= dwell_need:
                    hot.add(zname)
                    if key not in alerted:
                        alerted.add(key)
                        stats["alerts"] += 1
                        events.append({"time_s": round(now, 2), "track_id": p.tid, "zone": key[1], "event": "ALERT"})
                        if snap_dir:
                            snap = frame.copy()
                            draw_box(snap, p.box, RED, f"ALERT {key[1]}", s)
                            cv2.imwrite(os.path.join(snap_dir, f"alert_{stats['alerts']:02d}_t{now:05.1f}s.jpg"), snap)
                col = RED if dwell >= dwell_need else AMBER
                draw_box(frame, p.box, col, f"person #{p.tid} {dwell:.1f}s", s * 0.85, thickness=max(2, int(3 * s)))
            else:
                for key in [k for k in inside_since if k[0] == p.tid]:
                    events.append({"time_s": round(now, 2), "track_id": p.tid, "zone": key[1],
                                   "event": f"left after {now - inside_since.pop(key):.1f}s"})
                draw_box(frame, p.box, GREEN, f"person #{p.tid}", s * 0.85)
            cv2.circle(frame, tuple(p.foot.astype(int)), max(3, int(4 * s)), WHITE, -1, cv2.LINE_AA)
        stats["max_in_zone"] = max(stats["max_in_zone"], in_zone_now)

        # draw zones under a pulse when they are hot
        pulse = 0.25 + 0.2 * (np.sin(idx * 0.6) + 1) / 2
        for v, z in all_zones:
            draw_zone(frame, z, alpha=pulse if z.name in hot else 0.15, color=RED if z.name in hot else z.color, scale=s * 0.9)
            if v is not None:
                draw_box(frame, v.box, AMBER, f"{COCO[v.cls]} #{v.tid}", s * 0.8)
        status = ("ALERT: person in zone", RED) if hot else ("zone clear", GREEN)
        rows = [("SAFETY ZONES", AMBER), status, f"in zone now {in_zone_now:>3d}", f"alerts      {stats['alerts']:>3d}",
                (f"{job.fps:5.1f} FPS  OpenCV 5", (160, 160, 160))]
        draw_panel(frame, rows, (int(12 * s), int(12 * s)), s * 0.9)
        if hot:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), RED, max(4, int(8 * s)))
        return frame

    job.run(process)
    if cfg.get("csv"):
        _write_csv(cfg["csv"], events, ["time_s", "track_id", "zone", "event"])
    return {"events": events, "stats": stats, "fps": job.fps, "frames": len(job.proc_times)}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
