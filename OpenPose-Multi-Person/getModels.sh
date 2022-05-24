# ------------------------- BODY KEYPOINT-DETECTION MODELS -------------------------
# Downloading body pose (COCO)
OPENPOSE_URL="https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel"

OPENPOSE_FOLDER="pose/coco/"
wget -c ${OPENPOSE_URL} -P ${OPENPOSE_FOLDER}
