import cv2
import time 


def get_filtered_bboxes(img, min_area_ratio=0.001):
    """
    Get bounding boxes after filtering smaller boxes.
    
    Args:
        img (array) : Single channel thresholded image.
        min_area_ratio (float) : Minimum permissible area ratio of bounding boxes (box_area/image_area). Default is 0.001.       
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours according to area, larger to smaller.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    # Container to store filtered bboxes.
    bboxes = []
    # Image area.
    im_area = img.shape[0] * img.shape[1]
    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        # Remove very small detections.
        if cnt_area > min_area_ratio * im_area:
            bboxes.append((x, y, x+w, y+h))
    
    return bboxes


def draw_annotations(img, bboxes, thickness=2, color=(0,255,0)):
    """
    Draw bounding boxes around the objects.
    Args:
        img (ndarray): Image array.
        bboxes (tuple) : Bounding box coordinates in the form (x1, y1, x2, y2).
        thickness (int) : Bounding box line thickness, default is 2.
        color (tuple) : RGB color code, default is green.
    
    Returns:
        img (ndarray) : Image with annotations.
    """
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)

    return annotations


#============================================INITIALIZATIONS============================================#
cap = cv2.VideoCapture('Media/stags.mp4')
ret, frame = cap.read()
height, width = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
ksize=3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

out = cv2.VideoWriter('annotated-stags.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (2*width, height))
#========================================================================================================#

fps_cont = []
thresh_fps_cont = []
morph_fps_cont = []
cont_an_cont = []

while True:
    t_ = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    t1 = time.time()
    morphed = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(morphed, kernel)
    t2 = time.time()  
    bboxes = get_filtered_bboxes(eroded, min_area_ratio=0.005)
    t3 = time.time()
    draw = draw_annotations(frame, bboxes)
    morphed = [morphed, morphed, morphed]
    morphed = cv2.merge(morphed)
    conc = cv2.hconcat([morphed, draw])
    t4 = time.time()

    t_thresh = t1 - t0
    t_thresh_fps = 1/(t_thresh)
    thresh_fps_cont.append(t_thresh_fps)
    thresh_avg_fps = int(sum(thresh_fps_cont)/len(thresh_fps_cont))

    t_morph = t2 - t1
    t_morph_fps = 1/t_morph
    morph_fps_cont.append(t_morph_fps)
    morph_avg_fps = int(sum(morph_fps_cont)/len(morph_fps_cont))

    t_cont = t3 - t2
    cont_an_fps = 1/t_cont
    cont_an_cont.append(cont_an_fps)
    cont_avg_fps = int(sum(cont_an_cont)/len(cont_an_cont))

    t_net = t4 - t_
    t_net_fps = 1/t_net 
    fps_cont.append(t_net_fps)
    avg_fps = int(sum(fps_cont)/len(fps_cont))

    cv2.putText(conc, f"FPS: {avg_fps}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
    cv2.putText(conc, f"Size: {width}x{height}", (width-130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
    cv2.putText(conc, f"Thresholding FPS: {thresh_avg_fps}", (15, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
    cv2.putText(conc, f"Morphological Op FPS: {morph_avg_fps}", (15, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
    cv2.putText(conc, f"Contour Analysis FPS: {cont_avg_fps}", (15, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
    out.write(conc)

    tshow = time.time()
    cv2.imshow('Annotations', conc)
    cv2.imshow('Morphed', morphed)
    t_end = time.time()
    key = cv2.waitKey(1)

    if key == ord('q'):
        break 
    tttt = time.time()
    print(tshow-t3)
    print(f"FPS: {1/(tshow-t3)}")


cap.release()
out.release()
cv2.destroyAllWindows()
