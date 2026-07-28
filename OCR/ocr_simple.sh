#!/usr/bin/env bash
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 IMAGE [LANGUAGE] [PSM]" >&2
  exit 2
fi

image_path=$1
language=${2:-eng}
page_segmentation_mode=${3:-6}

if [ ! -f "$image_path" ]; then
  echo "error: image not found: $image_path" >&2
  exit 2
fi

exec tesseract \
  "$image_path" \
  stdout \
  -l "$language" \
  --oem 1 \
  --psm "$page_segmentation_mode"
