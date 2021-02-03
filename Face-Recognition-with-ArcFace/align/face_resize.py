import os
import cv2
from tqdm import tqdm


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def process_image(img):

    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])

    return pad_img


def main(source_root):

    dest_root = "/media/pc/6T/jasonjzhao/data/MS-Celeb-1M_Resized"
    mkdir(dest_root)
    cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = cv2.imread(os.path.join(source_root, subfolder, image_name))
            if type(img) == type(None):
                print("damaged image %s, del it" % (img))
                os.remove(img)
                continue
            size = img.shape
            h, w = size[0], size[1]
            if max(w, h) > 512:
                img_pad = process_image(img)
            else:
                img_pad = img
            cv2.imwrite(os.path.join(dest_root, subfolder, image_name.split('.')[0] + '.jpg'), img_pad)


if __name__ == "__main__":
    min_side = 512
    main(source_root = "/media/pc/6T/jasonjzhao/data/MS-Celeb-1M/database/base")