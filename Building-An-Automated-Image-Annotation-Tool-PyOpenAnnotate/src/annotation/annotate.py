#!/usr/bin/env python3

import os
import cv2
import sys
import argparse
import numpy as np
from annotation import utils
from tools import visualize
from natsort import natsorted


#--------------------------------INITIALIZATIONS--------------------------------#
coord = (20, 20)
del_entries = []
# Boolean to control bbox drawing loop.
draw_box = False
remove_box = False
# Update image.
updated_img = None
clean_img = None
org_img = None
max_area = 0
reset = False
PADDING = 10
Toggle  = False
min_area_ratio = 0.000
manual_assert_boxes = []
#-------------------------------------------------------------------------------#


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', '--img',
        help='path to the images file directory'
    )
    parser.add_argument(
        '-vid', '--vid',
        help='path to the video file'
    )
    parser.add_argument(
        '-T', '--toggle-mask',
        dest='toggle',
        action='store_true',
        help='Toggle Threshold Mask'
    )
    parser.add_argument(
        '--resume',
        help='path to annotations/labels directory'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=3,
        help="Number of frames to skip."
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args()
    return args


def image_paths(dir):
    # Iterate through the images.
    images_path = os.listdir(dir)
    # Remove files other than images.
    updated_images_paths = []

    for file in images_path:
        if ('.jpg' in file) or ('.png' in file) or ('.jpeg' in file):
            updated_images_paths.append(file)
    # print(f"Test1: {updated_images_paths}")
    updated_images_paths = natsorted(updated_images_paths)
    # print(f"Test2: {updated_images_paths}")

    with open('names.txt', 'w') as f:
        for path in updated_images_paths:
            ln = [path, '\n']
            f.writelines(ln)

    return updated_images_paths


def get_init_bboxes(img):
    """
    Returns bounding box using contour analysis.
    """
    global max_area, min_area_ratio
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    sorted_cnt.remove(sorted_cnt[0])
    max_area = img.shape[0] * img.shape[1]
    bounding_rect_boxes = []

    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        if (min_area_ratio * max_area < cnt_area):
            x = x - PADDING
            y = y - PADDING
            x = 0 if x <0 else x
            y = 0 if y < 0 else y
            x = img.shape[1] if x > img.shape[1] else x
            y = img.shape[0] if y > img.shape[0] else y
            bounding_rect_boxes.append(((x, y), (x+w, y+h)))
    return bounding_rect_boxes


def draw_init_annotations(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
        # print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2, cv2.LINE_AA)


def get_coordinate(event, x, y, flags, params):
    global coord, tlc, draw_box, bboxes, remove_box, clean_img, del_entries

    if event == cv2.EVENT_MOUSEMOVE:
        # Current coordinate. Updated every instant with the cursor.
        coord = (x, y)
        
    if event == cv2.EVENT_LBUTTONDOWN:
        # Clicked point.
        tlc = (x, y)
        draw_box = True

    if event == cv2.EVENT_LBUTTONUP:
        draw_box = False
        # Modify the code to draw rectangles only when area is greater than 
        # a particular threshold.
        # Also don't draw 'point' rectangles.
        if tlc != coord:
            cv2.rectangle(clean_img, tlc, coord, (255,0,0), 2, cv2.LINE_AA)
        # Append the final bbox coordinates to a global list.
        # Also remove very very small annotations.
        area = utils.get_box_area(tlc, coord)
        if area > 0.0001 * max_area:
            bboxes.append((tlc, coord))
            manual_assert_boxes.append((tlc, coord))

    # Add logic to remove a particular bounding box of double clicked in that area.
    if event == cv2.EVENT_LBUTTONDBLCLK:
        remove_box = True
        # Update the bboxes container.
        hit_point = (x, y)
        for point in bboxes:
            x1, y1 = point[0][0], point[0][1]
            x2, y2 = point[1][0], point[1][1]

            # Arrange small to large. Swap variables if required.
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            if hit_point[0] in range(x1, x2) and hit_point[1] in range(y1, y2):
                del_entries.append(point)
                bboxes.remove(point)
                # print('removed!')
        # print('Updated Bboxes: \n', bboxes)

        clean_img = org_img # Check point.
                
        # Update the bboxes annotations.
        if len(bboxes) >= 1:
            for point in bboxes:
                cv2.rectangle(clean_img, point[0], point[1], (255,0,0), 2, cv2.LINE_AA)
                # print('Boxes have been redrawn! ', point)
        remove_box = False


def update_bboxes(bboxes, del_entries, manual):
    for deleted_box in del_entries:
        # Deleted box coordinates. Area increased by 10%.
        x1_del, y1_del = int(0.9*deleted_box[0][1]), int(0.9*deleted_box[0][1])
        x2_del, y2_del = int(1.1*deleted_box[1][0]), int(1.1*deleted_box[1][1])
        for box in bboxes:
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[1][0], box[1][1]
            # Check if the points are inside the deleted region.
            if (x1_del< x1 < x2_del) and (x1_del < x2 < x2_del) and (y1_del < y1 < y2_del) and (y1_del < y2 < y2_del):
                bboxes.remove(box)
        # Add manually drawn boxes as well, given that it is not from the deleted list.
        if len(manual) > 0:
            for manual_box in manual:
                if (manual_box not in bboxes) and (manual_box not in del_entries):
                    bboxes.append(manual_box)
    return bboxes


# Load images.
def main():
    global coord, tlc, draw_box, clean_img, org_img, min_area_ratio
    global remove_box, bboxes, del_entries, reset, Toggle, manual_assert_boxes
    
    args = parser_opt()

    if args.vid is not None:
        file_type = 'vid'
        VID_PATH = args.vid
        if not os.path.isfile(VID_PATH):
            print('Please enter correct path to the video file.')
            sys.exit()
        
    elif args.img is not None:
        file_type = 'img'
        IMAGES_DIR = args.img
        if not os.path.isdir(IMAGES_DIR):
            print('Please enter correct images directory path.')
            sys.exit()
    else:
        print('Please provide the path to the image folder or video file.')

    if file_type == 'img':
        file_path = IMAGES_DIR
        updated_images_paths = image_paths(file_path)
        if args.resume is not None:
            completed_images = natsorted(os.listdir(args.resume))
            completed_images_names = []

            for file in completed_images:
                completed_images_names.append(file.split('.')[0])
            
            updated_im_paths = []
            for source_file in updated_images_paths:
                if not source_file.split('.')[0] in completed_images_names:
                    updated_im_paths.append(source_file)

            updated_images_paths = updated_im_paths

    elif file_type == 'vid':
        file_path = VID_PATH
        if not os.path.exists('images'):
            # Delete existing images. Feature to be added.
            os.mkdir('images')
        loading_img = np.zeros([400, 640, 3], dtype=np.uint8)
        skip_count = args.skip
        cap = cv2.VideoCapture(file_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        i = 0
        count = 0
        while cap.isOpened():
            if count/skip_count == 0:
                ret, frame = cap.read()
                
                load = loading_img.copy()
                if not ret:
                    print('Unable to read frame')
                    break
                cv2.putText(load, f"Frames: {i} / {int(frame_count)}", 
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(load, f"Sequencing...", 
                    (260, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
                cv2.imshow('Images', load)
                cv2.imwrite('images/img-{}.jpg'.format(i), frame)
            key = cv2.waitKey(1)
            i += 1
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyWindow('Images')
        updated_images_paths = image_paths('./images')
        file_path = './images'
        print(f"Images Saved to {os.getcwd()}/images")

    # Named window for Trackbars.
    cv2.namedWindow('Annotate')
    cv2.createTrackbar('threshold', 'Annotate', 127, 255, utils.ignore)
    cv2.createTrackbar('minArea', 'Annotate', 5, 500, utils.ignore)
    num = 0
    while True:
        if num == len(updated_images_paths):
            print('Task Completed.')
            break

        img_path = os.path.join(file_path, updated_images_paths[num])
        img = cv2.imread(img_path)
        original_height = img.shape[0]
        original_width = img.shape[1]

        resized_image = utils.aspect_resize(img)
        current_height = resized_image.shape[0]
        current_width = resized_image.shape[1]

        aspect_h = original_height/ current_height
        aspect_w = original_width/current_width
        aspect_ratio = [aspect_h, aspect_w]

        # Add all side padding 20 px.
        prev_thresh = 127
        prev_min_area = 0.00
        
        while True:
            img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            img_gray_padded = cv2.copyMakeBorder(img_gray, PADDING, PADDING, PADDING, PADDING, 
            	cv2.BORDER_CONSTANT, None, value=255)
            im_annotate = resized_image.copy()

            # Get trackbar threshold value.
            thresh_val = cv2.getTrackbarPos('threshold', 'Annotate')
            min_area_ratio = cv2.getTrackbarPos('minArea', 'Annotate')
            min_area_ratio = min_area_ratio*(1/10000)

            ret, thresh = cv2.threshold(img_gray_padded, thresh_val, 255, cv2.THRESH_BINARY)

            # The primary thresh image will be used to adjust thresholding when required.
            primary_thresh = thresh

            # Store the original image, might require later.
            org_img = im_annotate

            if clean_img is None:
                # Find contours and draw bounding rects.
                bboxes = get_init_bboxes(thresh)
                bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)

            # If threshold slider is moved, update bounding rects.
            elif (clean_img is not None) and (prev_thresh != thresh_val):
                reset = False
                bboxes = get_init_bboxes(thresh)
                bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)
                # print('Check : ', del_entries)
            elif (clean_img is not None) and prev_min_area != min_area_ratio:
                bboxes = get_init_bboxes(thresh)
                bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)

            else:
                # Update the thresh image if annotation performed once.
                im_annotate = clean_img
            
            clean_img = im_annotate.copy()
            prev_thresh = thresh_val
            prev_min_area = min_area_ratio

            draw_init_annotations(im_annotate, bboxes)
                
            cv2.setMouseCallback('Annotate', get_coordinate)

            h,w = im_annotate.shape[:2]
            horizontal_pt1 = (0, coord[1])
            horizontal_pt2 = (w, coord[1])
            vertical_pt1 = (coord[0], 0)
            vertical_pt2 = (coord[0], h)

            utils.draw_dotted_lines(im_annotate, horizontal_pt1, horizontal_pt2, (0,0,200))
            utils.draw_dotted_lines(im_annotate, vertical_pt1, vertical_pt2, (0,0,200))

            if draw_box:
                cv2.rectangle(im_annotate, tlc, coord, (255,0,0), 2, cv2.LINE_AA)

            if reset:
                im_annotate = org_img
      
            if args.toggle or Toggle:
                cv2.imshow('Mask', thresh)
            cv2.imshow('Annotate', im_annotate)
            # print(f"Org : {im_annotate.shape}, Thresh: {thresh.shape}")

            key = cv2.waitKey(1)
            # Store current threshold trackbar value to a temporary variable.
            thresh_val_prev = thresh_val
            # Press n to go to the next image.
            if key == ord('n') or key == ord('d'):
                clean_img = None
                utils.save(updated_images_paths[num].split('.')[0], (h, w), bboxes, aspect_ratio)
                num += 1
                bboxes = []
                del_entries = []
                manual_assert_boxes = []
                # print(f"Annotations Saved to {os.getcwd()}/labels")
                break
                
            if key == ord('b') or key == ord('a'):
                # print('Back Key Pressed.')
                # Go back one step.
                clean_img = None
                utils.save(updated_images_paths[num].split('.')[0], (h, w), bboxes, aspect_ratio)
                if num != 0:
                    num -= 1
                # print(f"Annotations Saved to {os.getcwd()}/labels")
                bboxes = []
                del_entries = []
                manual_assert_boxes = []
                break

            if key == ord('c'):
                reset = not reset
                utils.save(updated_images_paths[num].split('.')[0], (h, w), bboxes, aspect_ratio)
                bboxes = []
                del_entries = []
                manual_assert_boxes = []
            
            if key == ord('t'):
                Toggle = not Toggle
                if Toggle == False:
                    try:
                        cv2.destroyWindow('Mask')
                    except:
                        pass
                
            if key == ord('q'):
                print(f"Annotations Saved to {os.getcwd()}/labels")
                sys.exit()
    print(f"Annotations Saved to {os.getcwd()}/labels")

if __name__ == '__main__':
    main()
    
