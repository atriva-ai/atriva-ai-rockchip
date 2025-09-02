from ultralytics import YOLO
from yolov8_onnx import DetectEngine

# Export your model
model = YOLO("yolov8n.pt")
model.export(format="onnx", simplify=True, dynamic=True)

# Use it
engine = DetectEngine(model_path="yolov8n.onnx", image_size=640, conf_thres=0.5)
# engine(image) -> detection results
