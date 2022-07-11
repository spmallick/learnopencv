import cv2
import numpy as np
from core.config import cfg
import os
import random
import colorsys
from core.dependencies import *
import core.utils as utils

class Object_Detection:
    def __init__(self,model,allowed_classes,frame):
        
        Conf_threshold, NMS_threshold = 0.4,0.4
        self.classes, self.scores, self.boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        self.allowed_classes = allowed_classes
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    def read_class_names(class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names
    
    def draw_bbox(self,image,counted_classes = None):
        num_classes = len(self.class_names)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i in range(len(self.classes)):
            if int(self.classes[i]) < 0 or int(self.classes[i]) > num_classes: continue
            x,y,w,h = self.boxes[i]
            fontScale = 1
            score = self.scores[i]
            class_index = int(self.classes[i])                      
            class_name = self.class_names[int(class_index)]
            if class_name not in self.allowed_classes:
                continue
            else:
                bbox_color = colors[class_index]
                bbox_thick = int(0.6 * (image_h + image_w) / 600)
                c1, c2 = (x, y), (x + w, y + h)
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0),bbox_thick // 2, lineType=cv2.LINE_AA)

                if counted_classes != None:
                    height_ratio = int(image_h / 25)
                    offset = 15
                    for key, value in counted_classes.items():
                        cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                        offset += height_ratio
        return image

    def count_objects(self,by_class = False):
        
        counts = dict()
        if by_class:
            
            for i in range (len(self.classes)):
                
                class_index = int(self.classes[i])                      
                class_name = self.class_names[int(class_index)]

                if class_name in self.allowed_classes:
                    counts[class_name] = counts.get(class_name, 0) + 1
                else:
                    continue
        else:
            counts['total object'] = len(self.classes)
        return counts
    
    def crop_objects(self,frame,path):

        frame_copy = frame.copy()
        
        counts = dict()
        for i in range (len(self.classes)):
                
            class_index = int(self.classes[i])                      
            class_name = self.class_names[int(class_index)]

            if class_name in self.allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
                x, y, w,h = self.boxes[i]                
                try:
                    os.mkdir(path)
                except FileExistsError:
                    pass
                
                cropped_img = frame_copy[y:( y+ h)-5,x:(x+w)-5,]
                img_name = class_name + '_' +  str(counts[class_name]) + '.png'
                img_path = os.path.join(path, img_name)
                try:
                    cv2.imwrite(img_path, cropped_img)
                except:
                    pass
            else:
                continue
