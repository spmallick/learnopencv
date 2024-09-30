import matplotlib.pyplot as plt
import numpy as np
# from anue_labels import labels
# from scipy.misc import imread
import glob
from argparse import ArgumentParser
import random
import time
import cv2

import sys
sys.path.append("/home/somusan/OpencvUni/opencvblog/robotics-series/yolop_idd/public-code/helpers")
from anue_labels import labels, name2label
from annotation import Annotation

def get_level_id(label, level):
    if level == 3:
        return label.id
    elif level == 2:
        return label.level3Id
    elif level == 1:
        return label.level2Id
    elif level == 0:
        return label.level1Id
    else:
        return label.id

def get_ids(label, level):
    id_list = []
    for l in labels:
        if get_level_id(l, level) == label:
            id_list.append(l.id)
    return id_list

num_labels = [7, 16, 26, 35]

colors = [
    [(128, 64, 128), (244, 35, 232), (220, 20, 60), (0, 0, 230), (220, 190, 40), (70, 70, 70), (70, 130, 180), (0, 0, 0)], 
    [(128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140), (220, 20, 60), (255, 0, 0), (0, 0, 230), (255, 204, 54), (0, 0, 70), (220, 190, 40), (190, 153, 153), (174, 64, 67), (153, 153, 153), (70, 70, 70), (107, 142, 35), (70, 130, 180),(0, 0, 0)], 
    [(128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140), (220, 20, 60), (255, 0, 0), (0, 0, 230), (119, 11, 32), (255, 204, 54), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (220, 190, 40), (102, 102, 156), (190, 153, 153), (180, 165, 180), (174, 64, 67), (220, 220, 0), (250, 170, 30), (153, 153, 153), (169, 187, 214), (70, 70, 70), (150, 100, 100), (107, 142, 35), (70, 130, 180), (0, 0, 0)], 
    [(128, 64, 128), (250, 170, 160), (81, 0, 81), (244, 35, 232), (230, 150, 140), (152, 251, 152), (220, 20, 60), (246, 198, 145), (255, 0, 0), (0, 0, 230), (119, 11, 32), (255, 204, 54), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (136, 143, 153), (220, 190, 40), (102, 102, 156), (190, 153, 153), (180, 165, 180), (174, 64, 67), (220, 220, 0), (250, 170, 30), (153, 153, 153), (153, 153, 153), (169, 187, 214), (70, 70, 70), (150, 100, 100), (150, 120, 90), (107, 142, 35), (70, 130, 180), (169, 187, 214), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 142)]]


def get_image(label_mask, level):
    # print(label_mask.shape)
    h, w = label_mask.shape
    image = np.zeros((h,w,3), dtype=int)
    for l in range(num_labels[level]):
        id_list = get_ids(l, level)
        # print(id_list)
        for id in id_list:
            indices = label_mask == id 
            
            for i in range(3):
                
                image[indices,i] = colors[level][l][i]

    return image





def view_image(label_path):

    image_path = label_path.replace('gtFine', 'leftImg8bit').replace('_labelids','')
    label_mask = cv2.imread(label_path,0)
    image = cv2.imread(image_path,0)
    imgs = [image]
    for i in range(4):
        imgs.append(get_image(label_mask,4-i-1))


    f = plt.figure(figsize=(14,9), dpi=150)
            
    axarr = [None]*5
    axarr[0] = f.add_subplot(231)
    axarr[1] = f.add_subplot(232)
    axarr[2] = f.add_subplot(233)
    axarr[3] = f.add_subplot(234)
    axarr[4] = f.add_subplot(235)
    
    plt.subplots_adjust(top=0.9,
                        bottom=0.0,
                        left=0.0,
                        right=1.0,
                        hspace=0.15,
                        wspace=0.0)

    # f.
    # print(images[0].shape)
    id_names = [ 'image','id', 'level3', 'level2', 'level1'  ]
    for i in range(5):
        if imgs[i] is not None:
            axarr[i].imshow(imgs[i])
            axarr[i].axis('off')
            axarr[i].set_title(id_names[i])
    

    f.suptitle(label_path)
    plt.show()
    time.sleep(5)
    plt.close()




def get_args():
    parser = ArgumentParser()
    parser.add_argument('--datadir', default="")

    
    args = parser.parse_args()

    return args

# The main method
def main(args):
    label_path_list = glob.glob(args.datadir+f'/gtFine/*/*/*_labelids.png')
    print(len(label_path_list))
    random.shuffle(label_path_list)
    for l in label_path_list:
        
        view_image(l)




if __name__ == "__main__":
    args = get_args()
    main(args)




