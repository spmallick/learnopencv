"""
get_media.py
Download every clip and photo used by the examples from the LearnOpenCV GitHub repository
into the `assets` folder next to `code/` (created if missing). Paths are built with
os.path.join, so the same layout works on Windows, Linux, macOS and Colab.

All videos and photos are free to use under the Pexels license (https://www.pexels.com/license/),
except worker_zone.mp4, which is Intel's sample video (CC BY 4.0). They are provided for
educational purposes only, not for resale.
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")

# one place to change if the repository moves
GITHUB_ASSETS = "https://raw.githubusercontent.com/spmallick/learnopencv/master/opencv5-yolo26-use-cases/assets/"

CLIPS = [
    "traffic_highway_istanbul.mp4", "traffic_street_kathmandu.mp4", "traffic_suburban_lanes.mp4",
    "parking_supermarket.mp4", "parking_rooftop.mp4", "parking_rows_clip.mp4", "parking_dusk_clip.mp4",
    "rail_platform.mp4", "warehouse_forklift.mp4", "worker_zone.mp4",
]
PHOTOS = [
    "highway_lanes.jpg", "street_mixed_traffic.jpg", "parking_rows_still.jpg", "parking_dusk_still.jpg",
    "parking_topdown_still.jpg", "parking_aerial_angled.jpg", "platform_yellow_line.jpg", "warehouse_aisle.jpg",
]


def fetch(name, dst_dir=ASSETS):
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("  have", name)
        return dst
    print("  get ", name)
    req = urllib.request.Request(GITHUB_ASSETS + name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    if len(data) < 1024:                      # a broken upload (e.g. a 2-byte placeholder), not a real media file
        raise RuntimeError(f"{name}: only {len(data)} bytes came back from GitHub, the file there is not valid")
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def get_all(dst_dir=ASSETS):
    os.makedirs(dst_dir, exist_ok=True)
    print("assets ->", dst_dir)
    for name in CLIPS + PHOTOS:
        fetch(name, dst_dir)
    print("done:", len(CLIPS), "clips and", len(PHOTOS), "photos")


if __name__ == "__main__":
    get_all()
