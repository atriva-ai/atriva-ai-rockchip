import os
import shutil
import requests
import zipfile
from openvino.runtime import Core
from config import MODEL_DIR, MODEL_URLS

# Define OpenVINO-compatible accelerators
ACCELERATORS = ["cpui8", "cpu16", "cpu32"]

# Mapping user-friendly model names to actual OpenVINO models
MODEL_NAME_MAPPING = {
    "person": "person-detection-retail-0013",
    "face": "face-detection-retail-0005"
}

# OpenVINO Inference Engine Initialization
class ModelManager:
    BASE_DIR = MODEL_DIR

    def __init__(self, acceleration="cpu32"):
        if acceleration not in ACCELERATORS:
            raise ValueError(f"❌ Unsupported accelerator: {acceleration}")
        
        self.ie = Core()  # OpenVINO Inference Engine
        # Debug: Print available devices
        devices = self.ie.available_devices
        print(f"Available OpenVINO devices: {devices}")

        self.acceleration = acceleration
        self.MODEL_DIR = os.path.join(self.BASE_DIR, acceleration)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def download_model(self, model_name):
        """Download model files (XML & BIN) and save them properly."""
        model_url = MODEL_URLS.get(model_name, {}).get(self.acceleration)
        if not model_url:
            raise ValueError(f"❌ Model {model_name} not found for {self.acceleration}")

        model_folder = os.path.join(self.MODEL_DIR, model_name)
        os.makedirs(model_folder, exist_ok=True)

        xml_url = f"{model_url}{model_name}.xml"
        bin_url = f"{model_url}{model_name}.bin"

        for file_url in [xml_url, bin_url]:
            file_name = os.path.basename(file_url)
            file_path = os.path.join(model_folder, file_name)

            # ✅ Download the model safely
            response = requests.get(file_url, stream=True)
            content_type = response.headers.get("Content-Type", "")

            if response.status_code == 200 and "text/html" not in content_type:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"✅ Downloaded {file_name} to {file_path}")
            else:
                raise Exception(f"❌ Failed to download {file_name}. Server returned {response.status_code} ({content_type})")

        print(f"✅ Model {model_name} is ready for {self.acceleration} in {model_folder}")

    def load_model(self, requested_model):
        """Load a model using its friendly name or actual name."""
        model_name = MODEL_NAME_MAPPING.get(requested_model, requested_model)
        if model_name not in MODEL_URLS:
            raise ValueError(f"❌ Unknown model: {requested_model}. Available: {list(MODEL_NAME_MAPPING.keys())}")

        """Load a model from local storage, or download if missing."""
        model_folder = os.path.join(self.MODEL_DIR, model_name)
        xml_path = os.path.join(model_folder, f"{model_name}.xml")
        bin_path = os.path.join(model_folder, f"{model_name}.bin")

        # ✅ Check if the model already exists
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"✅ Model {model_name} found locally in {model_folder}")
             # Load the network model (XML & BIN)
            model = self.ie.read_model(model=xml_path)

            # Compile the model for inference
            compiled_model = self.ie.compile_model(model=model, device_name="CPU")  # Change to "GPU" or "MYRIAD" if needed
            return compiled_model

        print(f"⚠️ Model {model_name} not found locally. Downloading...")
        self.download_model(model_name)

        # ✅ Check again after downloading
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            print(f"✅ Model {model_name} is now ready in {model_folder}")

            # Load the network model (XML & BIN)
            model = self.ie.read_model(model=xml_path)

            # Compile the model for inference
            compiled_model = self.ie.compile_model(model=model, device_name="CPU")  # Change to "GPU" or "MYRIAD" if needed
            print(f"Model compiled successfully on device: CPU")

            input_shape = compiled_model.input(0).shape  # Expected shape (N, C, H, W)
            return compiled_model, input_shape
        else:
            raise Exception(f"❌ Failed to download {model_name}.")

    def list_models(self):
        """List all available models for each accelerator."""
        model_dict = {}

        for acc in ACCELERATORS:
            acc_dir = os.path.join(self.BASE_DIR, acc)
            if os.path.exists(acc_dir):
                model_dict[acc] = [
                    model for model in os.listdir(acc_dir) 
                    if os.path.isdir(os.path.join(acc_dir, model))
                ]
            else:
                model_dict[acc] = []  # No models found for this accelerator

        return model_dict

# Global model manager instance
model_manager = ModelManager(acceleration="cpu32")