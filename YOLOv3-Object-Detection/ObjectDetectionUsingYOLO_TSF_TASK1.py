import cv2
import numpy as np

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size = (len(classes), 3))

#Load image
img = cv2.imread("Sampleimage_2.jpg")
#img = cv2.resize(img, None, fx = 2.7, fy = 2.6)
height, width, channels = img.shape

#Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)

'''for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)'''

#Print the blobs detected
net.setInput(blob)
outs = net.forward(output_layers)
#print(outs)

#Showing informations on the screen
confidences = list()
boxes = list()
class_ids = list()

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            
            #Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #Rectangle Coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), 2)

#print(len(boxes))
#number_objects_detected = len(boxes)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()