import os

# Base directory for storing models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Predefined models with download URLs (modify as needed)
MODEL_URLS = {
    "person-detection-retail-0013": {
        "cpui8": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16-INT8/",
        "cpu16": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16/",
        "cpu32": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP32/"
    },
    "face-detection-retail-0005": {
        "cpui8": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP16-INT8/",
        "cpu16": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP16/",
        "cpu32": "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP32/"
    }
}