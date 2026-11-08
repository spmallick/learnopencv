"""
examples.py
The example configs used in the post. Run one with:
    python examples.py traffic_highway
or everything with:
    python examples.py all
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from usecases import count_traffic, monitor_parking, zone_alert  # noqa: E402

ROOT = os.path.dirname(HERE)


def M(f):
    return os.path.join(ROOT, "assets", f)


def O(f):
    return os.path.join(ROOT, "outputs", f)


MODELS = os.path.join(ROOT, "models")
VEH = ["car", "truck", "bus", "motorcycle"]

EXAMPLES = {
    # ---------------------------------------------------------------- traffic counting
    "traffic_highway": dict(
        kind="traffic", video=M("traffic_highway_istanbul.mp4"), out=O("01_traffic_highway.mp4"),
        csv=O("01_traffic_highway.csv"),
        model="yolo26n_640.onnx", imgsz=640, conf=0.3, classes=VEH, every=1, align=True,
        lines=[dict(name="Main road", a=(0.24, 0.62), b=(0.665, 0.62), labels=("towards", "away"),
                    classes=["car", "truck", "bus", "motorcycle"]),
               dict(name="Side road", a=(0.71, 0.55), b=(1.0, 0.55), labels=("towards", "away"),
                    classes=["car", "truck", "bus", "motorcycle"]),
               dict(name="Metrobus", a=(0.30, 0.25), b=(0.33, 0.50), labels=("east", "west"), classes=["bus"])],
    ),
    "traffic_street": dict(
        kind="traffic", video=M("traffic_street_kathmandu.mp4"), out=O("02_traffic_street.mp4"),
        csv=O("02_traffic_street.csv"),
        model="yolo26n_960.onnx", imgsz=960, conf=0.3, classes=["car", "truck", "bus", "motorcycle", "bicycle", "person"],
        every=2, merge_riders=True,
        lines=[dict(name="Road", a=(0.12, 0.62), b=(0.66, 0.62), labels=("towards", "away"),
                    classes=["car", "truck", "bus", "motorcycle", "bicycle"]),
               dict(name="Sidewalk", a=(0.66, 0.60), b=(0.99, 0.60), labels=("towards", "away"), anchor="foot",
                    classes=["person"])],
    ),
    "traffic_lanes": dict(
        kind="traffic", video=M("traffic_suburban_lanes.mp4"), out=O("03_traffic_lanes.mp4"), csv=O("03_traffic_lanes.csv"),
        model="yolo26n_640.onnx", imgsz=640, conf=0.3, classes=VEH, every=1,
        lines=[dict(name="Lane 1", a=(0.10, 0.72), b=(0.24, 0.72), labels=("towards", "away")),
               dict(name="Lane 2", a=(0.24, 0.72), b=(0.375, 0.72), labels=("towards", "away")),
               dict(name="Lane 3", a=(0.44, 0.72), b=(0.61, 0.72), labels=("towards", "away")),
               dict(name="Lane 4", a=(0.61, 0.72), b=(0.78, 0.72), labels=("towards", "away"))],
    ),
    # ---------------------------------------------------------------- parking
    "parking_supermarket": dict(
        kind="parking", video=M("parking_supermarket.mp4"), out=O("04_parking_supermarket.mp4"),
        chart=O("04_parking_occupancy.png"),
        model="yolo26n_960.onnx", imgsz=960, conf=0.25, every=1, align=True, park_speed=0.025,
        areas=[dict(name="West rows", pts=[(0.0, 0.52), (0.20, 0.51), (0.345, 0.60), (0.21, 0.78), (0.0, 0.87)]),
               dict(name="Center", pts=[(0.375, 0.61), (0.52, 0.62), (0.44, 1.0), (0.22, 1.0)]),
               dict(name="Store front", pts=[(0.55, 0.66), (0.73, 0.65), (0.70, 1.0), (0.50, 1.0)]),
               dict(name="East bays", pts=[(0.82, 0.70), (1.0, 0.70), (1.0, 1.0), (0.82, 1.0)])],
    ),
    # ---------------------------------------------------------------- zone alerts
    "zone_platform": dict(
        kind="zone", video=M("rail_platform.mp4"), out=O("05_zone_platform.mp4"), csv=O("05_zone_platform.csv"),
        snapshots=O("05_platform_alerts"), model="yolo26n_960.onnx", imgsz=960, conf=0.35, every=1, dwell_s=0.5,
        max_seconds=20,
        zones=[dict(name="Platform edge (west)", pts=[(0.415, 0.45), (0.445, 0.45), (0.29, 1.0), (0.215, 1.0)]),
               dict(name="Platform edge (east)", pts=[(0.565, 0.45), (0.59, 0.45), (0.77, 1.0), (0.70, 1.0)])],
    ),
    "zone_factory": dict(
        kind="zone", video=M("worker_zone.mp4"), out=O("06_zone_factory.mp4"), csv=O("06_zone_factory.csv"),
        snapshots=O("06_factory_alerts"), model="yolo26n_640.onnx", imgsz=640, conf=0.4, every=2, out_width=1280,
        max_seconds=40,
        dwell_s=1.0,
        zones=[dict(name="Machine area", pts=[(0.36, 0.73), (0.62, 0.565), (0.93, 0.78), (0.80, 1.0), (0.46, 1.0)])],
    ),
    "zone_forklift": dict(
        kind="zone", video=M("warehouse_forklift.mp4"), out=O("07_zone_forklift.mp4"), csv=O("07_zone_forklift.csv"),
        snapshots=O("07_forklift_alerts"), model="yolo26n_640.onnx", imgsz=640, conf=0.35, every=1, dwell_s=0.0,
        bubble=dict(cls="truck", margin=0.6, name="forklift bubble"),
    ),
}


def run(name, max_seconds=None):
    cfg = dict(EXAMPLES[name])
    cfg["model"] = os.path.join(MODELS, cfg["model"])
    if max_seconds:
        cfg["max_seconds"] = max_seconds
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    print(f"\n=== {name} ===")
    t = time.time()
    fn = {"traffic": count_traffic, "parking": monitor_parking, "zone": zone_alert}[cfg.pop("kind")]
    res = fn(cfg)
    res.pop("timeline", None)
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, default=str)[:600])
    print(f"events: {len(res.get('events', []))}   wall {time.time() - t:.1f}s")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else None
    for n in (EXAMPLES if which == "all" else which.split(",")):
        run(n, secs)
