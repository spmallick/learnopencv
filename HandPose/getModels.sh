# ------------------------- BODY, FACE AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
HAND_FOLDER="hand/"

# "------------------------- HAND MODELS -------------------------"
# Hand
HAND_MODEL=$HAND_FOLDER"pose_iter_102000.caffemodel"
wget -c ${OPENPOSE_URL}${HAND_MODEL} -P ${HAND_FOLDER}
