import os
import cv2 
import numpy as np


def ignore(x):
    pass


def save(img_file, shape, boxes, aspect_ratio):
    """
    Saves annotations to a text file in YOLO format,
    class, x_centre, y_centre, width, height
    """
    height_f = aspect_ratio[0]
    width_f = aspect_ratio[1]
    img_height = int(shape[0]*height_f)
    img_width = int(shape[1]*width_f)
    # print('Check : ', height_f, 'Width : ', width_f)

    # Check if the Annotations folder is empty.

    if not os.path.exists('labels'):
        os.mkdir('labels')
        
    with open('labels/' + img_file + '.txt', 'w') as f:
        for box in boxes:
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[1][0], box[1][1]
            # Map to the original image size.
            x1 = int(width_f*x1)
            y1 = int(height_f*y1)
            x2 = int(width_f*x2)
            y2 = int(height_f*y2)

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x_centre = int(x1 + width/2)
            y_centre = int(y1 + height/2)

            norm_xc = float(x_centre/img_width)
            norm_yc = float(y_centre/img_height)
            norm_width = float(width/img_width)
            norm_height = float(height/img_height)

            yolo_annotations = ['0', ' ' + str(norm_xc), ' ' + str(norm_yc), ' ' + str(norm_width), ' ' + str(norm_height), '\n']
            f.writelines(yolo_annotations)



def get_box_area(tlc, brc):
    x1, y1, x2, y2 = tlc[0], tlc[1], brc[0], brc[1]
    area = abs(x2 - x1) * abs(y2 - y1)
    return area


def draw_dotted_lines(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    """
    Draw dotted lines. 
    Adopted from StackOverflow.
    """
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5
    pts= []
    for i in  np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0] * (1-r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1-r) + pt2[1] * r) + 0.5)
        p = (x,y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i%2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def aspect_resize(img):
    prev_h, prev_w = img.shape[:2]
    # print(prev_h, prev_w)

    if prev_w > prev_h:
        current_w = 960
        aspect_ratio = current_w/prev_w
        current_h = int(aspect_ratio*prev_h)

    elif prev_w < prev_h:
        current_h = 720
        aspect_ratio = current_h/prev_h
        current_w = int(aspect_ratio*prev_h)

    else:
        if prev_h != 720:
            current_h, current_w = 720, 720

    res = cv2.resize(img, (current_w, current_h))
    return res