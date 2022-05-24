import time


# Correct the bounding box
def correct_bb(bb):
    bb.xmin = max(0, bb.xmin)
    bb.ymin = max(0, bb.ymin)
    bb.xmax = min(bb.xmax, 1)
    bb.ymax = min(bb.ymax, 1)
    return bb


# Main loop
while True:
    time.sleep(0.001)

    # Get image frame
    img = node.io['frame'].get()

    # Get detection output
    face_dets = node.io['face_det_in'].tryGet()
    if face_dets and img is not None:

        # Loop over all detections
        for det in face_dets.detections:

            # Correct bounding box
            correct_bb(det)
            node.warn(f"New detection {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")

            # Set config parameters
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(96, 112)  # Input size of Face Rec model
            cfg.setKeepAspectRatio(False)

            # Output image and config
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
