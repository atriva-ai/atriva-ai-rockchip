from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # This will download the model if missing
model.info()
