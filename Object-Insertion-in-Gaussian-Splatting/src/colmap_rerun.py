#!/usr/bin/env python3
"""Example of using Rerun to log and visualize the output of COLMAP's sparse reconstruction."""

from __future__ import annotations

import io
import os
import re
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import requests
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
from tqdm import tqdm

from read_write_model import Camera, read_model  # type: ignore[attr-defined]

DESCRIPTION = """
# Sparse reconstruction by COLMAP
This example was generated from the output of a sparse reconstruction done with COLMAP.

[COLMAP](https://colmap.github.io/index.html) is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline with a graphical and command-line interface.

In this example a short video clip has been processed offline by the COLMAP pipeline, and we use Rerun to visualize the
individual camera frames, estimated camera poses, and resulting point clouds over time.

The full source code for this example is available
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/structure_from_motion).
""".strip()

DATASET_DIR: Final = Path(__file__).parent.parent / "dataset"
DATASET_URL_BASE: Final = "https://storage.googleapis.com/rerun-example-datasets/colmap"
# When dataset filtering is turned on, drop views with less than this many valid points.
FILTER_MIN_VISIBLE: Final = 500


def scale_camera(camera: Camera, resize: tuple[int, int]) -> tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    assert camera.model == "PINHOLE"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(camera.params[:2] * scale_factor, camera.params[2:] * scale_factor)

    return (Camera(camera.id, camera.model, new_width, new_height, new_params), scale_factor)


def get_downloaded_dataset_path(dataset_name: str) -> Path:
    dataset_url = f"{DATASET_URL_BASE}/{dataset_name}.zip"

    recording_dir = DATASET_DIR / dataset_name
    if recording_dir.exists():
        return recording_dir

    os.makedirs(DATASET_DIR, exist_ok=True)

    zip_file = download_with_progress(dataset_url)

    with zipfile.ZipFile(zip_file) as zip_ref:
        progress = tqdm(zip_ref.infolist(), "Extracting dataset", total=len(zip_ref.infolist()), unit="files")
        for file in progress:
            zip_ref.extract(file, DATASET_DIR)
            progress.update()

    return recording_dir


def download_with_progress(url: str) -> io.BytesIO:
    """Download file with tqdm progress bar."""
    chunk_size = 1024 * 1024
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("content-length", 0))
    with tqdm(desc="Downloading dataset", total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as progress:
        zip_file = io.BytesIO()
        for data in resp.iter_content(chunk_size):
            zip_file.write(data)
            progress.update(len(data))

    zip_file.seek(0)
    return zip_file


def read_and_log_sparse_reconstruction(dataset_path: Path, filter_output: bool, resize: tuple[int, int] | None) -> None:
    print("Reading sparse COLMAP reconstruction")
    cameras, images, points3D = read_model(dataset_path / "sparse" / "0", ext=".bin")
    print("Building visualization by logging to Rerun")

    if filter_output:
        # Filter out noisy points
        points3D = {id: point for id, point in points3D.items() if point.rgb.any() and len(point.image_ids) > 4}

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), static=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("plot/avg_reproj_err", rr.SeriesLine(color=[240, 45, 58]), static=True)

    # Iterate through images (video frames) logging data related to each frame.
    for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
        image_file = dataset_path / "images" / image.name

        if not os.path.exists(image_file):
            continue

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        if resize:
            camera, scale_factor = scale_camera(camera, resize)
        else:
            scale_factor = np.array([1.0, 1.0])

        visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]
        visible_ids = image.point3D_ids[visible]

        if filter_output and len(visible_ids) < FILTER_MIN_VISIBLE:
            continue

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]
        if resize:
            visible_xys *= scale_factor

        rr.set_time_sequence("frame", frame_idx)

        points = [point.xyz for point in visible_xyzs]
        point_colors = [point.rgb for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        rr.log("plot/avg_reproj_err", rr.Scalar(np.mean(point_errors)))

        rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))

        # COLMAP's camera transform is "camera from world"
        rr.log(
            "camera", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True)
        )
        rr.log("camera", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward

        # Log camera intrinsics
        assert camera.model == "PINHOLE"
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=camera.params[:2],
                principal_point=camera.params[2:],
            ),
        )

        if resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rr.log("camera/image", rr.Image(bgr, color_model="BGR").compress(jpeg_quality=75))
        else:
            rr.log("camera/image", rr.EncodedImage(path=dataset_path / "images" / image.name))

        rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))


def main() -> None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    parser.add_argument("--unfiltered", action="store_true", help="If set, we don't filter away any noisy data.")
    parser.add_argument(
        "--dataset",
        action="store",
        default="colmap_rusty_car",
        choices=["colmap_rusty_car", "colmap_fiat"],
        help="Which dataset to download",
    )
    parser.add_argument("--resize", action="store", help="Target resolution to resize images")
    rr.script_add_args(parser)
    args = parser.parse_args()

    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split("x"))

    blueprint = rrb.Vertical(
        rrb.Spatial3DView(name="3D", origin="/",
                    # Set the background color to light blue.
                    # background=[100, 149, 237],
                    # Configure the line grid.
                    line_grid=rrb.archetypes.LineGrid3D(
                        visible=False,  # The grid is enabled by default, but you can hide it with this property.
                    ),
        ),
        
        rrb.Vertical(
            rrb.TextDocumentView(name="README", origin="/description"),
            rrb.Spatial2DView(name="Camera", origin="/camera/image"),
            rrb.TimeSeriesView(origin="/plot"),
        ),
        row_shares=[3, 2],
        collapse_panels=True,
    )

    rr.script_setup(args, "rerun_example_structure_from_motion", default_blueprint=blueprint)
    dataset_path = Path("/home/opencvuniv/Work/somusan/3dgs/dataset/360_v2/garden") # get_downloaded_dataset_path(args.dataset)
    read_and_log_sparse_reconstruction(dataset_path, filter_output=not args.unfiltered, resize=args.resize)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()