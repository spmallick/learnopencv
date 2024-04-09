import cv2
import numpy as np
import gradio as gr  # type: ignore
from mbnet import load_model, detect_objects, get_box_dimensions, draw_labels, load_img
from yolov3 import load_image, load_yolo, detect_objects_yolo, get_box_dimensions_yolo, draw_labels_yolo


# Image Inference


def img_inf(img, model):
    if model == "MobileNet-SSD":
        model, classes, colors = load_model()
        image, height, width, channels = load_img(img)
        blob, outputs = detect_objects(image, model)
        boxes, class_ids = get_box_dimensions(outputs, height, width)
        image1 = draw_labels(boxes, colors, class_ids, classes, image)
        return cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    else:
        model, classes, colors, output_layers = load_yolo()
        image, height, width, channels = load_image(img)
        blob, outputs = detect_objects_yolo(image, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions_yolo(outputs, height, width)
        image = draw_labels_yolo(boxes, confs, colors, class_ids, classes, image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


model_name = gr.Radio(["MobileNet-SSD", "YOLOv3"], value="YOLOv3", label="Model", info="choose your model")
inputs_image = gr.Image(type="filepath", label="Input Image")
outputs_image = gr.Image(type="numpy", label="Output Image")
interface_image = gr.Interface(
    fn=img_inf,
    inputs=[inputs_image, model_name],
    outputs=outputs_image,
    title="Image Inference",
    description="Upload your photo and select one model and see the results!",
    examples=[["sample/dog.jpg"]],
    cache_examples=False,
)


# Video Inference


def vid_inf(vid, model_type):
    if model_type == "MobileNet-SSD":
        cap = cv2.VideoCapture(vid)
        # get the video frames' width and height for proper saving of videos
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = "output_recorded.mp4"

        # create the `VideoWriter()` object
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        model, classes, colors = load_model()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model)
                boxes, class_ids = get_box_dimensions(outputs, height, width)
                frame = draw_labels(boxes, colors, class_ids, classes, frame)
                out.write(frame)
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        yield None, output_video

    else:
        cap = cv2.VideoCapture(vid)
        # get the video frames' width and height for proper saving of videos
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = "output_recorded.mp4"

        # create the `VideoWriter()` object
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

        model, classes, colors, output_layers = load_yolo()
        while cap.isOpened():
            ret, frame_y = cap.read()
            if ret:
                height, width, channels = frame_y.shape
                blob, outputs = detect_objects_yolo(frame_y, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions_yolo(outputs, height, width)
                frame_y = draw_labels_yolo(boxes, confs, colors, class_ids, classes, frame_y)
                out.write(frame_y)
                yield cv2.cvtColor(frame_y, cv2.COLOR_BGR2RGB), None
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        yield None, output_video


model_name = gr.Radio(["MobileNet-SSD", "YOLOv3"], value="YOLOv3", label="Model", info="choose your model")
input_video = gr.Video(sources=None, label="Input Video")
output_frame = gr.Image(type="numpy", label="Output Frames")
output_video_file = gr.Video(label="Output video")


interface_video = gr.Interface(
    fn=vid_inf,
    inputs=[input_video, model_name],
    outputs=[output_frame, output_video_file],
    title="Video Inference",
    description="Upload your video and select one model and see the results!",
    examples=[["sample/video_1.mp4"], ["sample/person.mp4"]],
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image, interface_video], tab_names=["Image", "Video"], title="GradioxOpenCV-DNN"
).queue().launch()
