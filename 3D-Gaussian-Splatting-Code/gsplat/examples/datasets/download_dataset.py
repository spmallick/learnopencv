"""Script to download benchmark dataset(s)"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

# dataset names
dataset_names = Literal[
    "mipnerf360",
    "mipnerf360_extra",
    "bilarf_data",
    "zipnerf",
    "zipnerf_undistorted",
]

# dataset urls
urls = {
    "mipnerf360": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "mipnerf360_extra": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
    "bilarf_data": "https://huggingface.co/datasets/Yuehao/bilarf_data/resolve/main/bilarf_data.zip",
    "zipnerf": [
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/berlin.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/london.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/nyc.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/alameda.zip",
    ],
    "zipnerf_undistorted": [
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
    ],
}

# rename maps
dataset_rename_map = {
    "mipnerf360": "360_v2",
    "mipnerf360_extra": "360_v2",
    "bilarf_data": "bilarf",
    "zipnerf": "zipnerf",
    "zipnerf_undistorted": "zipnerf_undistorted",
}


@dataclass
class DownloadData:
    dataset: dataset_names = "mipnerf360"
    save_dir: Path = Path(os.getcwd() + "/data")

    def main(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_download(self.dataset)

    def dataset_download(self, dataset: dataset_names):
        if isinstance(urls[dataset], list):
            for url in urls[dataset]:
                url_file_name = Path(url).name
                extract_path = self.save_dir / dataset_rename_map[dataset]
                download_path = extract_path / url_file_name
                download_and_extract(url, download_path, extract_path)
        else:
            url = urls[dataset]
            url_file_name = Path(url).name
            extract_path = self.save_dir / dataset_rename_map[dataset]
            download_path = extract_path / url_file_name
            download_and_extract(url, download_path, extract_path)


def download_and_extract(url: str, download_path: Path, extract_path: Path) -> None:
    download_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    # download
    download_command = [
        "curl",
        "-L",
        "-o",
        str(download_path),
        url,
    ]
    try:
        subprocess.run(download_command, check=True)
        print("File file downloaded succesfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")

    # if .zip
    if Path(url).suffix == ".zip":
        if os.name == "nt":  # Windows doesn't have 'unzip' but 'tar' works
            extract_command = [
                "tar",
                "-xvf",
                download_path,
                "-C",
                extract_path,
            ]
        else:
            extract_command = [
                "unzip",
                download_path,
                "-d",
                extract_path,
            ]
    # if .tar
    else:
        extract_command = [
            "tar",
            "-xvzf",
            download_path,
            "-C",
            extract_path,
        ]

    # extract
    try:
        subprocess.run(extract_command, check=True)
        os.remove(download_path)
        print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    tyro.cli(DownloadData).main()
