import os
import shutil
import subprocess
import requests
import zipfile
from pathlib import Path
from config import MODEL_DIR

# Define RKNN-compatible accelerators
ACCELERATORS = ["rk3588"]

# Mapping user-friendly model names to download URLs
MODEL_NAME_MAPPING = {
    'yolov8n': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx',
    'yolov8n-pose': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8_pose/yolov8n-pose.onnx',
    'yolov8n-obb': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8_obb/yolov8n-obb.onnx',
    'yolov11n': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11n.onnx',
    'LPRNet': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/LPRNet/lprnet.onnx',
    'RetinaFace-mobile-320': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_mobile320.onnx',
    'RetinaFace-res50-320': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_resnet50_320.onnx',
    'clip-images': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/clip/clip_images.onnx',
    'clip-text': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/clip/clip_text.onnx',
    'yamnet-3s': 'https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yamnet/yamnet_3s.onnx',    
}

# RKNN Model Manager
class ModelManager:
    BASE_DIR = "/tmp/models"  # Use writable temp directory for container

    def __init__(self, acceleration="rk3588"):
        if acceleration not in ACCELERATORS:
            raise ValueError(f"❌ Unsupported accelerator: {acceleration}")
        
        self.acceleration = acceleration
        self.MODEL_DIR = os.path.join(self.BASE_DIR, acceleration)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def download_model(self, model_name):
        """Download model using wget and store in /app/models/<model_name> folder."""
        if model_name not in MODEL_NAME_MAPPING:
            raise ValueError(f"❌ Unknown model: {model_name}. Available: {list(MODEL_NAME_MAPPING.keys())}")

        model_url = MODEL_NAME_MAPPING[model_name]
        model_folder = os.path.join(self.MODEL_DIR, model_name)
        
        # Create model folder
        try:
            os.makedirs(model_folder, exist_ok=True)
            print(f"📁 Created model folder: {model_folder}")
        except Exception as e:
            raise Exception(f"❌ Failed to create model folder {model_folder}: {str(e)}")
        
        # Extract filename from URL
        filename = os.path.basename(model_url)
        if not filename:
            filename = f"{model_name}.onnx"
        
        file_path = os.path.join(model_folder, filename)
        
        # Check if model already exists
        if os.path.exists(file_path):
            print(f"✅ Model {model_name} already exists at {file_path}")
            return file_path
        
        print(f"📥 Downloading {model_name} from {model_url}")
        print(f"📁 Saving to {file_path}")
        
        try:
            # Use wget to download the model
            cmd = [
                "wget",
                "--no-verbose",  # Quiet output
                "--show-progress",  # Show progress bar
                "--timeout=300",  # 5 minute timeout
                "--tries=3",  # Retry 3 times
                "--output-document", file_path,  # Save to specific file
                model_url
            ]
            
            print(f"🔧 Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ wget command completed successfully")
            
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"✅ Successfully downloaded {model_name} to {file_path}")
                print(f"📊 File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
                return file_path
            else:
                raise Exception(f"❌ Download completed but file is missing or empty: {file_path}")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ Failed to download {model_name}: {e.stderr}"
            print(error_msg)
            print(f"wget stdout: {e.stdout}")
            print(f"wget stderr: {e.stderr}")
            print(f"wget return code: {e.returncode}")
            # Clean up partial download
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"❌ Unexpected error downloading {model_name}: {str(e)}"
            print(error_msg)
            # Clean up partial download
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(error_msg)

    def convert_onnx_to_rknn(self, onnx_path, rknn_path, model_name):
        """Convert ONNX model to RKNN format with quantization."""
        try:
            print(f"🔄 Starting RKNN conversion for {model_name}")
            
            # Check if RKNN toolkit is available
            try:
                from rknn.api import RKNN
                print(f"✅ RKNN toolkit imported successfully")
            except ImportError as e:
                raise Exception(f"❌ RKNN toolkit not available: {str(e)}")
            
            print(f"🔄 Converting {model_name} from ONNX to RKNN...")
            print(f"📁 ONNX: {onnx_path}")
            print(f"📁 RKNN: {rknn_path}")
            
            # Check if ONNX file exists
            if not os.path.exists(onnx_path):
                raise Exception(f"❌ ONNX file not found: {onnx_path}")
            
            # Initialize RKNN
            rknn = RKNN(verbose=True)
            
            # Configure preprocessing and target platform FIRST
            print(f"⚙️ Configuring model for RK3588...")
            ret = rknn.config(
                target_platform='rk3588',
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]]
            )
            if ret != 0:
                raise Exception(f"❌ Failed to configure model: {ret}")
            print(f"✅ Model configured successfully")
            
            # Load ONNX model AFTER configuration
            print(f"📥 Loading ONNX model: {onnx_path}")
            ret = rknn.load_onnx(model=onnx_path)
            if ret != 0:
                raise Exception(f"❌ Failed to load ONNX model: {ret}")
            print(f"✅ ONNX model loaded successfully")
            
            # Build the model without quantization (faster, no dataset required)
            print(f"🔨 Building RKNN model without quantization...")
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise Exception(f"❌ Failed to build RKNN model: {ret}")
            print(f"✅ RKNN model built successfully")
            
            # Export to RKNN file
            print(f"💾 Exporting RKNN model: {rknn_path}")
            ret = rknn.export_rknn(rknn_path)
            if ret != 0:
                raise Exception(f"❌ Failed to export RKNN model: {ret}")
            print(f"✅ RKNN model exported successfully")
            
            # Release resources
            rknn.release()
            print(f"🧹 RKNN resources released")
            
            if os.path.exists(rknn_path) and os.path.getsize(rknn_path) > 0:
                print(f"✅ Successfully converted {model_name} to RKNN format")
                print(f"📊 RKNN file size: {os.path.getsize(rknn_path) / (1024*1024):.2f} MB")
                return rknn_path
            else:
                raise Exception(f"❌ RKNN file was not created or is empty: {rknn_path}")
                
        except ImportError:
            raise Exception("❌ RKNN toolkit not available. Please install rknn-toolkit2")
        except Exception as e:
            error_msg = f"❌ Failed to convert {model_name} to RKNN: {str(e)}"
            print(error_msg)
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Clean up partial RKNN file
            if os.path.exists(rknn_path):
                os.remove(rknn_path)
            raise Exception(error_msg)

    def load_model(self, requested_model):
        """Load a model using its friendly name, download if missing, convert to RKNN if needed."""
        # Check if the requested model exists in our mapping
        if requested_model not in MODEL_NAME_MAPPING:
            raise ValueError(f"❌ Unknown model: {requested_model}. Available: {list(MODEL_NAME_MAPPING.keys())}")

        model_name = requested_model  # Use the requested model name directly
        model_folder = os.path.join(self.MODEL_DIR, model_name)
        
        # Check if RKNN model exists (preferred for inference)
        rknn_path = os.path.join(model_folder, f"{model_name}.rknn")
        if os.path.exists(rknn_path):
            print(f"✅ RKNN model {model_name} found at {rknn_path}")
            return rknn_path
        
        # Check if ONNX model exists
        onnx_files = [f for f in os.listdir(model_folder) if f.endswith('.onnx')] if os.path.exists(model_folder) else []
        
        if onnx_files:
            onnx_file = onnx_files[0]
            onnx_path = os.path.join(model_folder, onnx_file)
            print(f"⚠️ ONNX model found but RKNN missing. Converting {model_name} to RKNN format...")
            return self.convert_onnx_to_rknn(onnx_path, rknn_path, model_name)
        else:
            print(f"⚠️ Model {model_name} not found locally. Downloading...")
            onnx_path = self.download_model(model_name)
            print(f"⚠️ Converting downloaded {model_name} to RKNN format...")
            return self.convert_onnx_to_rknn(onnx_path, rknn_path, model_name)

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

    def get_model_info(self, model_name):
        """Get information about a specific model."""
        if model_name not in MODEL_NAME_MAPPING:
            return {"error": f"Unknown model: {model_name}"}
        
        model_folder = os.path.join(self.MODEL_DIR, model_name)
        model_url = MODEL_NAME_MAPPING[model_name]
        
        info = {
            "name": model_name,
            "url": model_url,
            "local_path": model_folder,
            "onnx_exists": False,
            "rknn_exists": False,
            "onnx_size": 0,
            "rknn_size": 0
        }
        
        if os.path.exists(model_folder):
            # Check for ONNX file
            onnx_files = [f for f in os.listdir(model_folder) if f.endswith('.onnx')]
            if onnx_files:
                onnx_file = onnx_files[0]
                onnx_path = os.path.join(model_folder, onnx_file)
                info["onnx_exists"] = True
                info["onnx_size"] = os.path.getsize(onnx_path)
                info["onnx_filename"] = onnx_file
            
            # Check for RKNN file
            rknn_path = os.path.join(model_folder, f"{model_name}.rknn")
            if os.path.exists(rknn_path):
                info["rknn_exists"] = True
                info["rknn_size"] = os.path.getsize(rknn_path)
        
        return info

# Global model manager instance
model_manager = ModelManager(acceleration="rk3588")