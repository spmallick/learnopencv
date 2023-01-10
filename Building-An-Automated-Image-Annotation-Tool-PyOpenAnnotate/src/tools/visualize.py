import cv2
import os
import sys
import argparse
import glob 
from natsort import natsorted


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', '--img',
        required=True,
        help='path to the images file directory'
    )
    parser.add_argument(
        '--ann',
        required=True,
        help='path to the annotations directory'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='bounding box thickness'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args()
    return args


def read_annotations(img, annotation):
    height, width = img.shape[:2]
    bboxes = []
    with open(annotation, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name = line.replace('\n', '')
            class_id, xc, yc, w, h = name.split(' ')
            class_id = int(class_id)
            xc, yc = float(xc), float(yc)
            h, w = float(h), float(w)
            box_h = int(height*h)
            box_w = int(width*w)
            x_center = int(xc*width)
            y_center = int(yc*height)
            x1 = x_center - int(box_w/2)
            y1 = y_center - int(box_h/2)
            x2 = x1 + box_w 
            y2 = y1 + box_h
            p1 = (x1, y1)
            p2 = (x2, y2)
            bboxes.append((p1, p2))
    return bboxes


def draw_annotations(img, bboxes, thickness=2):
    p1 = bboxes[0]
    p2 = bboxes[1]
    cv2.rectangle(img, p1, p2, (0,255,0), thickness)
    return img


def main():
    args = parser_opt()

    multiple_img = False
    multiple_annotations = False

    img_path_prefix = args.img
    ann_path_prefix = args.ann
    thickness = args.thickness

    if not os.path.exists(args.img):
        print('Invalid Image or Image directory')
        sys.exit()

    if not os.path.exists(args.ann):
        print('Invalid Annotations Directory')
        sys.exit()

    if os.path.isdir(args.img):
        multiple_img = True

        if os.path.isfile(args.ann):
            print('Image and Annotation path type mismatch.')
            sys.exit()

        img_files = os.listdir(args.img)
        image_files_list = []
        for f in img_files:
            if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'):
                image_files_list.append(f)
       
        img_files = natsorted(image_files_list)

        if len(img_files) == 0:
            print('Empty Image Directory')
            sys.exit()

        if os.path.isdir(args.ann):
            multiple_annotations = True

            annotation_files = glob.glob(args.ann + '/*.txt')
            annotation_files = natsorted(annotation_files)

            if len(annotation_files) == 0:
                print('\n Empty Annotation directory.')
                print('\n Must contain text files with annotations in YOLO format.')
                sys.exit()
    n = 0
    while multiple_img and multiple_annotations:

        if n == len(annotation_files) or n == len(img_files):
            print('All annotations viewed.')
            break

        img_path = os.path.join(img_path_prefix, img_files[n])
        ann_path = os.path.join(ann_path_prefix, img_files[n].split(".")[0] + '.txt')

        img = cv2.imread(img_path)
        bounding_boxes = read_annotations(img, ann_path)
        annotated_img = draw_annotations(img, bounding_boxes, thickness)
        cv2.imshow(f"{img_files[n]}", annotated_img)
        key = cv2.waitKey(0)

        if key == ord('n') or key == ord('d'):
            cv2.destroyWindow(img_files[n])
            n += 1
        
        if key == ord('a') or key == ord('b'):
            cv2.destroyWindow(img_files[n])
            if n!= 0:
                n -= 1
        
        if key == ord('q'):
            break

    if os.path.isfile(args.img):
        if not os.path.isfile(args.ann):
            print('Image and Annotation path type mismatch.')
        
        ann_file_extension = args.ann.split(".")[-1]
        # print('Extension : ', ann_file_extension)
        if ann_file_extension != 'txt':
            print('\nInvalid annotation file')
            print('Please provide text file with annotations in YOLO format.')
        
        img = cv2.imread(args.img)
        bounding_boxes = read_annotations(img, args.ann)
        annotated_img = draw_annotations(img, bounding_boxes, thickness)

        cv2.imshow(f"{args.ann}", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()