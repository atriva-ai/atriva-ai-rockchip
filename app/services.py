import numpy as np
import cv2
import os
from app.models import model_manager

# Object to model mapping for object detection
OBJECT_MODEL_MAPPING = {
    # Person detection
    "human": "yolov8n",
    "person": "yolov8n",
    "people": "yolov8n",
    
    # Face detection
    "face": "RetinaFace-mobile-320",
    "faces": "RetinaFace-mobile-320",
    "person-face": "RetinaFace-mobile-320",
    
    # License plate detection
    "license-plate": "LPRNet",
    "license_plate": "LPRNet",
    "plate": "LPRNet",
    "lpr": "LPRNet",
    
    # Pose detection
    "pose": "yolov8n-pose",
    "human-pose": "yolov8n-pose",
    "person-pose": "yolov8n-pose",
    "keypoints": "yolov8n-pose",
    
    # Object detection (general)
    "object": "yolov8n",
    "objects": "yolov8n",
    "detection": "yolov8n",
    
    # Vehicle detection
    "vehicle": "yolov8n",
    "car": "yolov8n",
    "truck": "yolov8n",
    "bus": "yolov8n",
    
    # Animal detection
    "animal": "yolov8n",
    "dog": "yolov8n",
    "cat": "yolov8n",
    
    # CLIP models for image/text understanding
    "clip-image": "clip-images",
    "clip-text": "clip-text",
    "image-understanding": "clip-images",
    "text-understanding": "clip-text",
    
    # Audio models
    "audio": "yamnet-3s",
    "sound": "yamnet-3s",
    "speech": "yamnet-3s",
    
    # Oriented bounding box detection
    "obb": "yolov8n-obb",
    "oriented-object": "yolov8n-obb",
    "rotated-object": "yolov8n-obb",
    
    # YOLO11 for newer detection
    "yolo11": "yolov11n",
    "yolo11n": "yolov11n",
}

def preprocess_image(image_bytes: bytes, target_shape: tuple):
    """Preprocess image to match RKNN model input shape."""
    h, w = target_shape  # Extract height and width from model shape
    target_shape = (w, h)  # Model expects (width, height)

    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Resize image to match model input dimensions
    image_resized = cv2.resize(image, target_shape)
    print(f'Resize input image to {w} x {h} matching model size')

    # Convert image format from (H, W, C) -> (C, H, W)
    image_transposed = image_resized.transpose((2, 0, 1))  # Channels first

    # Normalize and add batch dimension
    image_transposed = np.expand_dims(image_transposed, axis=0).astype(np.float32)
    print(f"Preprocessed input shape: {image_transposed.shape}")  # Debugging

    return image_transposed

class RKNNInferenceManager:
    """Manages RKNN model loading and inference with runtime optimization."""
    
    def __init__(self):
        self.rknn = None
        self.model_path = None
        self.is_initialized = False
        
    def load_model(self, model_name):
        """Load RKNN model and initialize runtime."""
        try:
            from rknn.api import RKNN
            
            # Get model path from model manager
            model_path = model_manager.load_model(model_name)
            
            # If model path changed, reload the model
            if self.model_path != model_path:
                self.release()  # Clean up previous model
                
                print(f"🔄 Loading RKNN model: {model_path}")
                self.rknn = RKNN(verbose=True)
                
                # Load the RKNN model
                ret = self.rknn.load_rknn(model_path)
                if ret != 0:
                    raise Exception(f"❌ Failed to load RKNN model: {ret}")
                
                # Initialize runtime for RK3588
                print(f"⚙️ Initializing RKNN runtime for RK3588...")
                ret = self.rknn.init_runtime(target='rk3588')
                if ret != 0:
                    raise Exception(f"❌ Failed to initialize RKNN runtime: {ret}")
                
                self.model_path = model_path
                self.is_initialized = True
                print(f"✅ RKNN model loaded and runtime initialized successfully")
                
            return True
            
        except ImportError:
            raise Exception("❌ RKNN toolkit not available. Please install rknn-toolkit2")
        except Exception as e:
            error_msg = f"❌ Failed to load RKNN model: {str(e)}"
            print(error_msg)
            self.release()
            raise Exception(error_msg)
    
    def run_inference(self, input_data):
        """Run inference using loaded RKNN model."""
        if not self.is_initialized or self.rknn is None:
            raise Exception("❌ RKNN model not loaded. Call load_model() first.")
        
        try:
            print(f"🔍 Running inference with input shape: {input_data.shape}")
            
            # Run inference
            outputs = self.rknn.inference(inputs=[input_data])
            
            # RKNN returns a list of outputs, get the first one
            if outputs and len(outputs) > 0:
                output = outputs[0]
                print(f"✅ Inference completed. Output shape: {output.shape}")
                return output
            else:
                raise Exception("❌ No output received from RKNN inference")
                
        except Exception as e:
            error_msg = f"❌ Inference failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def release(self):
        """Release RKNN resources."""
        if self.rknn is not None:
            try:
                self.rknn.release()
                print("🧹 RKNN resources released")
            except Exception as e:
                print(f"⚠️ Warning: Error releasing RKNN resources: {e}")
            finally:
                self.rknn = None
                self.model_path = None
                self.is_initialized = False

# Global RKNN inference manager
rknn_manager = RKNNInferenceManager()

def run_inference(input_data, model_name):
    """Runs inference on input data using RKNN model.
    
    Args:
        input_data: Preprocessed input data
        model_name: Direct model name (e.g., 'yolov8n', 'LPRNet', etc.)
    
    Returns:
        Model output
    """
    try:
        # Load model if not already loaded
        rknn_manager.load_model(model_name)
        
        # Run inference
        output = rknn_manager.run_inference(input_data)
        
        # Remove batch dimension if present
        if len(output.shape) > 2:
            output = np.squeeze(output)
        
        return output
        
    except Exception as e:
        print(f"❌ Inference error: {str(e)}")
        raise e

def run_object_detection(image_bytes: bytes, object_name: str):
    """Runs object detection inference using RKNN model.
    
    Args:
        image_bytes: Raw image bytes
        object_name: Object to detect (e.g., 'human', 'face', 'license-plate', etc.)
                   This will be mapped to the appropriate model.
    
    Returns:
        List of detections
    """
    try:
        # Map object name to model name
        if object_name in OBJECT_MODEL_MAPPING:
            model_name = OBJECT_MODEL_MAPPING[object_name]
            print(f"🎯 Mapping '{object_name}' to model '{model_name}'")
        else:
            # If not found in mapping, try to use the object_name directly as model name
            model_name = object_name
            print(f"⚠️ No mapping found for '{object_name}', using as model name directly")
        
        # Get model info to determine input shape
        model_info = model_manager.get_model_info(model_name)
        if "error" in model_info:
            raise Exception(f"❌ Model not found: {model_name}")
        
        # Use standard input shape for object detection models
        # Most YOLO models expect 640x640 or similar
        input_shape = (640, 640)  # (height, width)
        
        # Preprocess image for model input
        image = preprocess_image(image_bytes, input_shape)
        
        # Run inference
        model_output = run_inference(image, model_name)
        
        # Set a confidence threshold
        confidence_threshold = 0.3
        detections = []
        
        # Parse model output (adjust based on actual model output format)
        if len(model_output.shape) == 2:  # (num_detections, 7) format
            for detection in model_output:
                if len(detection) >= 7:
                    image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
                    if confidence > confidence_threshold:
                        detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
        elif len(model_output.shape) == 3:  # (1, num_detections, 7) format
            for detection in model_output[0]:
                if len(detection) >= 7:
                    image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
                    if confidence > confidence_threshold:
                        detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
        
        # Print results
        print(f"🔍 Found {len(detections)} detections for '{object_name}':")
        for det in detections:
            print(f"  Class: {det['class_id']}, Confidence: {det['confidence']:.3f}, BBox: {det['bbox']}")
        
        return detections
        
    except Exception as e:
        print(f"❌ Object detection error: {str(e)}")
        raise e

def get_available_objects():
    """Get list of available object types for detection."""
    return list(OBJECT_MODEL_MAPPING.keys())

def get_available_models():
    """Get list of available model names for direct inference."""
    return list(model_manager.MODEL_NAME_MAPPING.keys())

def cleanup_rknn():
    """Cleanup RKNN resources when shutting down."""
    rknn_manager.release()