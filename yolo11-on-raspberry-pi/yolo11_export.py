from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Export the model
model.export(format="ncnn")
