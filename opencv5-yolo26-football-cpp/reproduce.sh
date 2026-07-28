#!/usr/bin/env bash
# Reproduce every annotated output shown in the blog, from the source media in
# assets/, using the demos built in build/. Results land in outputs/.
#
# Prereqs:
#   1. Build OpenCV 5  -> ./build_opencv5.sh
#   2. Build the demos -> cmake -G Ninja -B build -DOpenCV_DIR=.../opencv5_install/lib/cmake/opencv5
#                         cmake --build build
#   3. Run this script. It auto-exports the YOLO26 ONNX models the first time
#      (Ultralytics downloads the nano weights), then runs every demo.
#
# The models/ and outputs/ folders are generated here, never committed.
set -euo pipefail
PY=${PYTHON:-python3}
BIN=build
A=assets
M=models
O=outputs
mkdir -p "$O" "$M"

DET1280=$M/yolo26n_1280.onnx
DET640=$M/yolo26n_640.onnx
SEG=$M/yolo26n_seg_640.onnx
POSE=$M/yolo26n_pose_640.onnx

# --- auto-export the models on first run (needs python + ultralytics + onnx) ---
if [ ! -f "$DET640" ] || [ ! -f "$DET1280" ] || [ ! -f "$SEG" ] || [ ! -f "$POSE" ]; then
  echo "Models missing, exporting YOLO26 to ONNX (weights download automatically)..."
  "$PY" scripts/export_yolo26_onnx.py --imgsz 640
  "$PY" scripts/export_yolo26_onnx.py --imgsz 1280
  "$PY" scripts/export_variants.py            # yolo26n-seg + yolo26n-pose
fi
for model in "$DET640" "$DET1280" "$SEG" "$POSE"; do
    [ -s "$model" ] || {
        echo "model export did not create a nonempty file: $model" >&2
        exit 1
    }
done

echo "== Object detection (images) =="
"$BIN/detect_image" --model "$DET1280" --imgsz 1280 --engine new --conf 0.25 \
    --source "$A/images/team_lineup.jpg"   --out "$O/detection_germany.jpg"
"$BIN/detect_image" --model "$DET1280" --imgsz 1280 --engine new --conf 0.25 \
    --source "$A/images/match_broadcast.jpg"  --out "$O/detection_broadcast.jpg"
"$BIN/detect_image" --model "$DET1280" --imgsz 1280 --engine new --conf 0.25 \
    --source "$A/images/player_duel.jpg"             --out "$O/detection_duel_limitation.jpg"

echo "== Object detection (video) =="
"$BIN/detect_video" --model "$DET640" --imgsz 640 --engine new \
    --source "$A/videos/match_play.mp4"   --out "$O/detection_football.mp4"

echo "== Detection on wide stadium footage =="
"$BIN/detect_video" --model "$DET640" --imgsz 640 --engine new --conf 0.25 \
    --source "$A/videos/stadium_wide.mp4"      --out "$O/detection_stadium.mp4"

echo "== Team split (jersey colour) =="
"$BIN/detect_teams" --model "$DET640" --imgsz 640 --teams 2 --minh 0.08 \
    --source "$A/videos/stadium_wide.mp4"          --out "$O/team_split_stadium.mp4"

echo "== Pose / keypoints =="
"$BIN/detect_pose" --model "$POSE" --conf 0.30 \
    --source "$A/images/aerial_duel.jpg"      --out "$O/pose_header.jpg"
Y26_PERSON_LABEL=Dancer "$BIN/detect_pose" --model "$POSE" \
    --source "$A/videos/halftime_dancers.mp4"          --out "$O/pose_dancers.mp4"

echo "== Instance segmentation =="
"$BIN/detect_seg" --model "$SEG" --conf 0.35 \
    --source "$A/images/goal_celebration.jpg"      --out "$O/seg_celebration.jpg"
"$BIN/detect_seg" --model "$SEG" --conf 0.30 \
    --source "$A/images/player_duel.jpg"             --out "$O/seg_duel.jpg"
# the halftime performers read better as "dancer" than "player"
Y26_PERSON_LABEL=Dancer "$BIN/detect_seg" --model "$SEG" --conf 0.35 \
    --source "$A/videos/halftime_dancers.mp4"          --out "$O/seg_dancers.mp4"
"$BIN/detect_seg" --model "$SEG" --conf 0.35 \
    --source "$A/videos/ball_closeup.mp4"     --out "$O/seg_ball_closeup.mp4"

echo "== Web-safe re-encode =="
# OpenCV's VideoWriter emits MPEG-4 Part 2, which browsers will not play. Re-encode
# every output clip to H.264 (yuv420p) + faststart so the MP4s work on the web.
if command -v ffmpeg >/dev/null 2>&1; then
    for v in "$O"/*.mp4; do
        [ -e "$v" ] || continue
        tmp="${v%.mp4}.h264.mp4"
        ffmpeg -y -hide_banner -loglevel error -i "$v" \
            -c:v libx264 -pix_fmt yuv420p -crf 20 -preset medium -movflags +faststart -an "$tmp" \
            && mv -f "$tmp" "$v" && echo "  web-encoded $(basename "$v")"
    done
else
    echo "  ffmpeg not found; skipping (browser playback needs H.264, not MPEG-4)."
fi

echo
echo "Done. Annotated results are in $O/"
