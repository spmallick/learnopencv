from rfdetr import RFDETRBase
import supervision as sv

model = RFDETRBase(pretrain_weights = "./checkpoints/checkpoint_best_regular.pth")
# class_names = model.class_names
categories = [{"id": 0, "name": 'fish', "supercategory": "animal"}, {"id": 1, "name": 'jellyfish', "supercategory": "animal"}, {"id": 2, "name": "penguin", "supercategory": "animal"}, {"id": 3, "name": "puffing", "supercategory": "animal"}, {"id": 4, "name": "shark", "supercategory": "animal"}, {"id": 5, "name": "stingray", "supercategory": "animal"}, {"id": 6, "name": "starfish","supercategory": "animal"}]

def callback(frame, index):
    annotated_frame = frame.copy()

    detections = model.predict(annotated_frame, threshold = 0.6)

    labels = [
        f"{categories[class_id]['name']}  {confidence: .2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # print(tuple(zip(detections.class_id, detections.confidence)))
    
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame

sv.process_video(
    source_path = "./video_3.mp4",
    target_path = "./output_annotations_4.mp4",
    callback = callback,
)
