from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import imutils
import cv2
import numpy as np
import argparse,os
from keras.preprocessing import image
import matplotlib.pyplot as plt

# parameters for loading data and images
parser = argparse.ArgumentParser(description='Test Emotion Detection Model')
parser.add_argument('--image', dest='image', help='Use --image [Path to images directory]')
parser.add_argument('--result', dest='result', help='Use --result [Path to output directory]')
args = parser.parse_args()

detection_model_path = './detection/haarcascade_frontalface_default.xml'
emotion_model_path = './models/_mini_XCEPTION.132-0.66.hdf5'
image_path = args.image
result_dir = args.result
emotion_offsets = (20, 40)


# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]


# Pre-Processing the input

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# Drawing inferences

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),cv2.FONT_HERSHEY_SIMPLEX,font_scale, color, thickness, cv2.LINE_AA)

emotion_target_size = emotion_classifier.input_shape[1:3]

test_images = os.listdir(image_path)

for test_img in test_images:
    img_path = image_path + test_img 
    print(img_path)
    rgb_image = image.load_img(img_path, grayscale = False,target_size= None)
    rgb_image = image.img_to_array(rgb_image)
    gray_image = image.load_img(img_path, grayscale = True,target_size= None)
    gray_image = image.img_to_array(gray_image)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
    print(gray_image.shape)
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

    for face_coordinates in faces:
        x, y, width, height = face_coordinates
        x_off, y_off = emotion_offsets
        x1, x2, y1, y2 = x - x_off, x + width + x_off, y - y_off, y + height + y_off
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = EMOTIONS[emotion_label_arg]
        
        color = (255, 0, 0)
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_dir + 'result_' + test_img, bgr_image)
