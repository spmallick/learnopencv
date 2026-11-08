"""
still_images.py
Still-image examples, two or three per use case. Photos are free-license Pexels images
(ids in the README) plus frames pulled from three of the clips; all live in ../assets.

    traffic   per-lane vehicle counts on a highway photo; vehicle mix on a dense street
    parking   per-area counts on three lots (supermarket rows, dusk, near-overhead drone) and the top-down limit
    zones     people behind a platform's yellow line; workers in a forklift aisle
"""

import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocv5_det import COCO, Detector, draw_box, draw_panel, ui_scale  # noqa: E402
from usecases import count_image, parking_image, zone_image  # noqa: E402

ROOT = os.path.dirname(HERE)
IMG = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")

# six lanes of a divided highway, drawn as quads from the bottom edge up to where the road narrows
HIGHWAY_LANES = [
    dict(name="Lane 1 (to cam)", pts=[(0.0, 1.0), (0.15, 1.0), (0.39, 0.40), (0.33, 0.40)]),
    dict(name="Lane 2 (to cam)", pts=[(0.15, 1.0), (0.32, 1.0), (0.45, 0.40), (0.39, 0.40)]),
    dict(name="Lane 3 (to cam)", pts=[(0.32, 1.0), (0.48, 1.0), (0.50, 0.40), (0.45, 0.40)]),
    dict(name="Lane 4 (away)", pts=[(0.55, 1.0), (0.70, 1.0), (0.58, 0.40), (0.52, 0.40)]),
    dict(name="Lane 5 (away)", pts=[(0.70, 1.0), (0.86, 1.0), (0.64, 0.40), (0.58, 0.40)]),
    dict(name="Lane 6 (away)", pts=[(0.86, 1.0), (1.0, 1.0), (1.0, 0.72), (0.70, 0.40), (0.64, 0.40)]),
]
STREET_ROAD = [dict(name="Road", pts=[(0.0, 0.12), (1.0, 0.12), (1.0, 1.0), (0.0, 1.0)])]

ROWS_AREAS = [
    dict(name="Curb row", pts=[(0.47, 0.36), (0.53, 0.36), (0.62, 1.0), (0.41, 1.0)]),
    dict(name="Center rows", pts=[(0.62, 0.38), (0.79, 0.38), (0.85, 0.68), (0.71, 0.68)]),
    dict(name="East rows", pts=[(0.80, 0.37), (1.0, 0.37), (1.0, 1.0), (0.80, 1.0), (0.855, 0.68)]),
    dict(name="Service yard", pts=[(0.0, 0.42), (0.30, 0.30), (0.36, 0.32), (0.12, 0.68), (0.0, 0.68)]),
]
DUSK_AREAS = [
    dict(name="Left block", pts=[(0.03, 0.63), (0.28, 0.63), (0.30, 0.83), (0.0, 0.86)]),
    dict(name="Center block", pts=[(0.30, 0.62), (0.57, 0.63), (0.58, 0.85), (0.32, 0.84)]),
    dict(name="Right block", pts=[(0.58, 0.70), (0.74, 0.70), (0.74, 0.82), (0.58, 0.82)]),
    dict(name="Front row", pts=[(0.0, 0.86), (0.78, 0.84), (0.78, 1.0), (0.0, 1.0)]),
]
AERIAL_AREAS = [dict(name="Parking lot", pts=[(0.49, 0.42), (0.76, 0.33), (0.80, 0.50), (0.72, 0.93), (0.56, 0.93)])]

PLATFORM_ZONES = [dict(name="Behind the yellow line", pts=[(0.51, 0.44), (0.61, 0.44), (0.61, 1.0), (0.45, 1.0)])]
AISLE_ZONES = [dict(name="Forklift aisle", pts=[(0.44, 0.42), (0.56, 0.42), (0.78, 1.0), (0.22, 1.0)])]


def topdown_limit(det, path_in, path_out):
    """Run the COCO detector on a straight-down view and show what it thinks it sees."""
    img = cv2.imread(path_in)
    s = ui_scale(img)
    dets = det(img, conf=0.15)                      # low threshold on purpose: show everything it finds
    names = {}
    for d in dets:
        name = COCO[int(d[5])]
        names[name] = names.get(name, 0) + 1
        draw_box(img, d[:4], (0, 190, 255), f"{name} {d[4]:.2f}", s * 0.7)
    rows = [("TOP-DOWN VIEW (COCO model)", (0, 190, 255))]
    rows += [f"{k:<12s}{v:>3d}" for k, v in sorted(names.items(), key=lambda kv: -kv[1])] or ["nothing found"]
    draw_panel(img, rows, (int(12 * s), int(12 * s)), s * 0.9)
    cv2.imwrite(path_out, img)
    return names


def run_all(models=MODELS, img=IMG, out=OUT):
    os.makedirs(out, exist_ok=True)

    def p(f):
        return os.path.join(img, f)

    def o(f):
        return os.path.join(out, f)

    veh = Detector(os.path.join(models, "yolo26n_960.onnx"), 960, conf=0.3, classes=["car", "truck", "bus", "motorcycle"])
    mix = Detector(os.path.join(models, "yolo26n_960.onnx"), 960, conf=0.3,
                   classes=["car", "truck", "bus", "motorcycle", "bicycle", "person"])
    cars = Detector(os.path.join(models, "yolo26n_960.onnx"), 960, conf=0.25, classes=["car", "truck", "bus"])
    people = Detector(os.path.join(models, "yolo26n_960.onnx"), 960, conf=0.3, classes=["person"])
    res = {}

    # traffic
    c, outside, d = count_image(p("highway_lanes.jpg"), o("01b_highway_lanes_still.jpg"), HIGHWAY_LANES, veh,
                                tiles=(2, 3), title="VEHICLES PER LANE", label_pos="bottom")
    res["highway lanes"] = {k: sum(v.values()) for k, v in c.items()}
    print("highway lanes:", res["highway lanes"], "outside", outside)
    for tag, tiles in (("single", None), ("tiled", (2, 3))):
        c, outside, d = count_image(p("street_mixed_traffic.jpg"), o(f"02b_street_mix_still_{tag}.jpg"), STREET_ROAD,
                                    mix, tiles=tiles, merge=True, title="VEHICLE MIX")
        res[f"street mix {tag}"] = c["Road"]
        print(f"street mix {tag}:", c["Road"])

    # parking
    for tag, tiles in (("single", None), ("tiled", (2, 3))):
        c, n = parking_image(p("parking_rows_still.jpg"), o(f"04b_parking_still_{tag}.jpg"), ROWS_AREAS, cars, tiles)
        res[f"parking rows {tag}"] = (c, n)
        print(f"parking rows {tag}:", c, "total", n)
    c, n = parking_image(p("parking_dusk_still.jpg"), o("04d_parking_dusk_still.jpg"), DUSK_AREAS, cars, tiles=(2, 3))
    res["parking dusk"] = (c, n)
    print("parking dusk:", c, "total", n)
    for tag, tiles in (("single", None), ("tiled", (3, 4))):
        c, n = parking_image(p("parking_aerial_angled.jpg"), o(f"04e_parking_aerial_{tag}.jpg"), AERIAL_AREAS, cars, tiles)
        res[f"parking aerial {tag}"] = (c, n)
        print(f"parking aerial {tag}:", c, "total", n)
    res["top-down"] = topdown_limit(Detector(os.path.join(models, "yolo26n_960.onnx"), 960),
                                    p("parking_topdown_still.jpg"), o("04c_parking_topdown_limit.jpg"))
    print("top-down:", res["top-down"])

    # zones
    z, n = zone_image(p("platform_yellow_line.jpg"), o("05b_platform_still.jpg"), PLATFORM_ZONES, people, tiles=(2, 2))
    res["platform"] = (z, n)
    print("platform:", z, "people", n)
    z, n = zone_image(p("warehouse_aisle.jpg"), o("06b_warehouse_aisle_still.jpg"), AISLE_ZONES, people)
    res["aisle"] = (z, n)
    print("aisle:", z, "people", n)
    return res


if __name__ == "__main__":
    run_all()
