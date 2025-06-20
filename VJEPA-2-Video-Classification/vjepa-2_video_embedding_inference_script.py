import cv2
import torch
from collections import deque


# Constants
NUM_FRAMES = 7
FRAME_WIDTH, FRAME_HEIGHT = 256, 256


cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)


# Delay transformer imports until after imshow is initialized
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

# Load model and processor
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
model = VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2").to("cuda").eval()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

frame_buffer = deque(maxlen=NUM_FRAMES)
frame_count = 0

def preprocess_clip(frames):
    resized = [cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT)) for f in frames]
    return processor(resized, return_tensors="pt").to("cuda")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Show frame using OpenCV
        cv2.imshow("Display", frame)

        frame_count += 1

        frame_buffer.append(frame)

        if len(frame_buffer) == NUM_FRAMES:
            inputs = preprocess_clip(list(frame_buffer))
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            label = model.config.id2label[predicted_label]
            print(f"Prediction: {label}")
            frame_buffer.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:import cv2
