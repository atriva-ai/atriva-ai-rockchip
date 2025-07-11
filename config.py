import os

# Base directory for storing models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# RKNN model configuration
RKNN_CONFIG = {
    "target_platform": "rk3588",
    "mean_values": [[0, 0, 0]],
    "std_values": [[255, 255, 255]],
    "do_quantization": True
}

# Model input shapes for different model types
MODEL_INPUT_SHAPES = {
    "yolov8n": (640, 640),
    "yolov8n-pose": (640, 640),
    "yolov8n-obb": (640, 640),
    "yolov11n": (640, 640),
    "LPRNet": (320, 320),
    "RetinaFace-mobile-320": (320, 320),
    "RetinaFace-res50-320": (320, 320),
    "clip-images": (224, 224),
    "clip-text": (224, 224),
    "yamnet-3s": (96, 64)
}