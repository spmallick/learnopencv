import gradio as gr
import cv2
import requests
import os

from ultralytics import YOLO

file_urls = [
    'https://www.dropbox.com/s/b5g97xo901zb3ds/pothole_example.jpg?dl=1',
    'https://www.dropbox.com/s/86uxlxxlm1iaexa/pothole_screenshot.png?dl=1',
    'https://www.dropbox.com/s/7sjfwncffg8xej2/video_7.mp4?dl=1'
]

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(
            file_urls[i],
            f"video.mp4"
        )
    else:
        download_file(
            file_urls[i],
            f"image_{i}.jpg"
        )

model = YOLO('best.pt')
path  = [['image_0.jpg'], ['image_1.jpg']]
video_path = [['video.mp4']]

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Pothole detector",
    examples=path,
    cache_examples=False,
)

def show_preds_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            outputs = model.predict(source=frame)
            results = outputs[0].cpu().numpy()
            for i, det in enumerate(results.boxes.xyxy):
                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

inputs_video = [
    gr.components.Video(type="filepath", label="Input Video"),

]
outputs_video = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Pothole detector",
    examples=video_path,
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference']
).queue().launch()