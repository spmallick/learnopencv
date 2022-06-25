import cv2
import numpy as np
from core.config import cfg
import os
from deep_sort import preprocessing
from deep_sort.detection import Detection
import core.utils as utils

class Dependencies:

    def __init__(self,frame,allowed_classes):
        self.frame = frame
        self.allowed_classes = allowed_classes
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    def deep_sort(self):
        
        classes, scores, boxes = cfg.all_model.detect(self.frame, 0.4, 0.4)
        names = []
        deleted_indx = []
        for i in range(len(boxes)):
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            if class_name not in self.allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        bboxes = np.delete(boxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        features = cfg.encoder(self.frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, cfg.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        return detections

    def draw_functions(self,bbox,track,color):
        
        fontScale = 1
        image_h, image_w, _ = self.frame.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)

        cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, bbox_thick)

        bbox_mess = '%s' % (str(track))
        cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(bbox_mess))*17, int(bbox[1])), color, -1)
        cv2.putText(self.frame, str(bbox_mess),(int(bbox[0]), int(bbox[1]-10)),0, fontScale, (255,255,255),bbox_thick)
    

    def crop_image(self,track_id,bbox):
        
        frame_copy = self.frame.copy()
        cropped_obj = frame_copy[int(bbox[1]):int(bbox[3]),(int(bbox[0])):int(bbox[2])]
        self.new_cropped_obj = cropped_obj.copy()
        
        aplha = 1.5
        beta = 0
        adjust = cv2.convertScaleAbs(cropped_obj,alpha=aplha,beta=beta)
        nr = cv2.fastNlMeansDenoising(adjust)

        crop_path = os.path.join(os.getcwd(), 'detections', 'lp_detections')
        filename = str(track_id) +"vehicle.png"
        img_path = os.path.join(crop_path, filename)
        cv2.imwrite(img_path,cropped_obj)

        ratio = float(max(self.new_cropped_obj.shape[:2]))/min(self.new_cropped_obj.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
        
        return bound_dim,nr

    def save_crop_image(self,num_boxes,final_path):
        filename = "number" + str(len(num_boxes)) + ".png"
        img_path = os.path.join(final_path, filename)
        cv2.imwrite(img_path, self.new_cropped_obj)
