import cv2
import numpy as np

# import time


# video_path = 'D:/OfficeWork/VS_code_exp/exp/video_1.mp4'
# image_path = 'D:/OfficeWork/VS_code_exp/exp/test.jpg.jpg'


def load_model():
    model = cv2.dnn.readNet(
        model="frozen_inference_graph.pb", config="ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", framework="TensorFlow"
    )
    with open("object_detection_classes_coco.txt", "r") as f:
        class_names = f.read().split("\n")
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    return model, class_names, COLORS


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def detect_objects(img, net):
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104, 117, 123), swapRB=True)
    net.setInput(blob)
    outputs = net.forward()
    print(outputs.shape)
    # print (outputs)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    class_ids = []

    for detect in outputs[0, 0, :, :]:
        scores = detect[2]
        class_id = detect[1]
        if scores > 0.3:
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[5] * width)
            h = int(detect[6] * height)
            x = int((detect[3] * width))
            y = int((detect[4] * height))
            boxes.append([x, y, w, h])
            class_ids.append(class_id)
    return boxes, class_ids


def draw_labels(boxes, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    model, classes, colors = load_model()
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes[int(class_ids[0]) - 1]
        color = colors[i]
        cv2.rectangle(img, (x, y), (w, h), color, 5)
        cv2.putText(img, label, (x, y - 5), font, 5, color, 5)
    return img


def image_detect(img_path):
    model, classes, colors = load_model()
    image, height, width, channels = load_img(img_path)
    blob, outputs = detect_objects(image, model)
    boxes, class_ids = get_box_dimensions(outputs, height, width)
    image1 = draw_labels(boxes, colors, class_ids, classes, image)
    return image1

    # def start_video(video_path):
    model, classes, colors = load_model()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model)
        boxes, class_ids = get_box_dimensions(outputs, height, width)
        frame = draw_labels(boxes, colors, class_ids, classes, frame)
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


cv2.destroyAllWindows()
