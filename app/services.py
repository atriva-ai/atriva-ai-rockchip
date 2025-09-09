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
                # Try different initialization approaches
                try:
                    # First try with target specification
                    ret = self.rknn.init_runtime(target='rk3588')
                    if ret != 0:
                        print(f"⚠️ Target initialization failed (ret={ret}), trying CPU simulation mode...")
                        # Try CPU simulation mode
                        ret = self.rknn.init_runtime(target='rk3588', device_id='cpu')
                        if ret != 0:
                            print(f"⚠️ CPU simulation failed (ret={ret}), trying default...")
                            # Try default initialization
                            ret = self.rknn.init_runtime()
                            if ret != 0:
                                raise Exception(f"❌ All RKNN runtime initialization attempts failed: {ret}")
                except Exception as e:
                    print(f"⚠️ RKNN runtime initialization error: {str(e)}")
                    # Try CPU simulation mode as fallback
                    try:
                        print(f"🔄 Trying CPU simulation mode...")
                        ret = self.rknn.init_runtime(target='rk3588', device_id='cpu')
                        if ret != 0:
                            print(f"⚠️ CPU simulation also failed (ret={ret}), trying default...")
                            ret = self.rknn.init_runtime()
                            if ret != 0:
                                raise Exception(f"❌ All RKNN runtime initialization methods failed. Last error: {ret}")
                    except Exception as cpu_e:
                        print(f"⚠️ CPU simulation initialization also failed: {str(cpu_e)}")
                        # Last resort - try without any parameters
                        ret = self.rknn.init_runtime()
                        if ret != 0:
                            raise Exception(f"❌ All RKNN runtime initialization methods failed. Last error: {ret}")
                
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
            print(f"🔍 Input data type: {input_data.dtype}")
            print(f"🔍 Input data range: [{input_data.min():.3f}, {input_data.max():.3f}]")
            
            # Run inference
            print(f"⚡ Calling RKNN inference...")
            outputs = self.rknn.inference(inputs=[input_data])
            print(f"✅ RKNN inference call completed")
            
            # RKNN returns a list of outputs, get the first one
            if outputs and len(outputs) > 0:
                output = outputs[0]
                print(f"✅ Inference completed. Output shape: {output.shape}")
                print(f"🔍 Output data type: {output.dtype}")
                print(f"🔍 Output data range: [{output.min():.3f}, {output.max():.3f}]")
                return output
            else:
                raise Exception("❌ No output received from RKNN inference")
                
        except Exception as e:
            error_msg = f"❌ Inference failed: {str(e)}"
            print(error_msg)
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
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

def post_process_standard_yolo(output, confidence_threshold=0.3):
    """
    Post-process standard YOLOv8 output format (84, 8400).
    
    Args:
        output: YOLOv8 model output with shape (84, 8400)
               - First 4 rows: bbox coordinates (x_center, y_center, width, height)
               - Next 80 rows: class probabilities
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detections with class_id, confidence, and bbox
    """
    import numpy as np
    
    if len(output.shape) != 2 or output.shape[0] != 84:
        print(f"⚠️ Unexpected standard YOLO output shape: {output.shape}")
        return []
    
    # YOLOv8 uses 3 detection layers with different strides
    # The 8400 detections come from: 80x80 + 40x40 + 20x20 = 6400 + 1600 + 400 = 8400
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]
    
    # Extract bbox coordinates (first 4 rows)
    bbox_data = output[:4]  # Shape: (4, 8400)
    tx, ty, tw, th = bbox_data
    
    # Extract class probabilities (next 80 rows)
    class_probs = output[4:84]  # Shape: (80, 8400)
    
    detections = []
    
    # Process each detection
    detection_idx = 0
    for layer_idx, (stride, grid_size) in enumerate(zip(strides, grid_sizes)):
        for y in range(grid_size):
            for x in range(grid_size):
                if detection_idx >= output.shape[1]:
                    break
                
                # Get class probabilities for this detection
                class_scores = class_probs[:, detection_idx]
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                # Apply sigmoid to class confidence
                confidence = 1 / (1 + np.exp(-class_conf))
                
                if confidence > confidence_threshold:
                    # Decode YOLOv8 bbox coordinates
                    # Apply sigmoid to bbox coordinates
                    sigmoid_tx = 1 / (1 + np.exp(-tx[detection_idx]))
                    sigmoid_ty = 1 / (1 + np.exp(-ty[detection_idx]))
                    sigmoid_tw = 1 / (1 + np.exp(-tw[detection_idx]))
                    sigmoid_th = 1 / (1 + np.exp(-th[detection_idx]))
                    
                    # YOLOv8 decode formula
                    cx = (sigmoid_tx * 2 - 0.5 + x) * stride
                    cy = (sigmoid_ty * 2 - 0.5 + y) * stride
                    w = (sigmoid_tw * 2) ** 2 * stride
                    h = (sigmoid_th * 2) ** 2 * stride
                    
                    # Convert to corner coordinates
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2
                    
                    # Normalize to [0, 1] (assuming input image is 640x640)
                    xmin = xmin / 640.0
                    ymin = ymin / 640.0
                    xmax = xmax / 640.0
                    ymax = ymax / 640.0
                    
                    # Ensure coordinates are within [0, 1]
                    xmin = max(0, min(1, xmin))
                    ymin = max(0, min(1, ymin))
                    xmax = max(0, min(1, xmax))
                    ymax = max(0, min(1, ymax))
                    
                    # Debug: print some values to understand the format
                    if len(detections) < 5:  # Only print first 5 for debugging
                        print(f"🔍 Detection {len(detections)}: class={class_id}, conf={confidence:.3f}, bbox=[{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]")
                    
                    detections.append({
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                
                detection_idx += 1
    
    print(f"🔍 Processed {len(detections)} detections from standard YOLOv8 output")
    return detections

def post_process_yolo_output(output, confidence_threshold=0.3):
    """
    Post-process YOLO output feature map to extract detections.
    
    Args:
        output: YOLO model output with shape (1, channels, height, width)
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detections with class_id, confidence, and bbox
    """
    import numpy as np
    
    # Handle both 3D and 4D formats
    if len(output.shape) == 4:
        output = output[0]  # Shape: (channels, height, width)
    elif len(output.shape) == 3:
        # Already in (channels, height, width) format
        pass
    else:
        print(f"⚠️ Unexpected output shape: {output.shape}")
        return []
    
    channels, height, width = output.shape
    print(f"🔍 YOLO output: channels={channels}, height={height}, width={width}")
    
    detections = []
    
    # For YOLOv8, typically we have:
    # - 4 channels for bbox (x, y, w, h)
    # - 1 channel for objectness
    # - 80 channels for class probabilities (COCO classes)
    # Total: 85 channels
    
    if channels >= 85:  # Standard YOLOv8 format
        # Extract bbox coordinates (x, y, w, h)
        bbox_x = output[0]  # Shape: (height, width)
        bbox_y = output[1]  # Shape: (height, width)
        bbox_w = output[2]  # Shape: (height, width)
        bbox_h = output[3]  # Shape: (height, width)
        
        # Extract objectness score
        objectness = output[4]  # Shape: (height, width)
        
        # Extract class probabilities
        class_probs = output[5:85]  # Shape: (80, height, width)
        
        # Find detections above threshold
        for y in range(height):
            for x in range(width):
                obj_score = objectness[y, x]
                if obj_score > confidence_threshold:
                    # Get class probabilities for this location
                    class_scores = class_probs[:, y, x]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    # Combined confidence
                    confidence = obj_score * class_conf
                    
                    if confidence > confidence_threshold:
                        # Convert from grid coordinates to image coordinates
                        # YOLO outputs are normalized (0-1)
                        center_x = (x + bbox_x[y, x]) / width
                        center_y = (y + bbox_y[y, x]) / height
                        box_w = bbox_w[y, x] / width
                        box_h = bbox_h[y, x] / height
                        
                        # Convert to corner coordinates
                        xmin = center_x - box_w / 2
                        ymin = center_y - box_h / 2
                        xmax = center_x + box_w / 2
                        ymax = center_y + box_h / 2
                        
                        detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
    
    elif channels == 64:  # Custom 64-channel format
        print(f"🔍 Processing 64-channel YOLO format...")
        
        # Try format: 4 bbox + 1 objectness + 59 classes = 64 channels
        bbox_x = output[0]  # Shape: (height, width)
        bbox_y = output[1]  # Shape: (height, width)
        bbox_w = output[2]  # Shape: (height, width)
        bbox_h = output[3]  # Shape: (height, width)
        
        # Extract objectness score
        objectness = output[4]  # Shape: (height, width)
        
        # Extract class probabilities (remaining 60 channels)
        class_probs = output[5:]  # Shape: (59, height, width)
        
        num_classes = len(class_probs)
        print(f"🔍 Processing {num_classes} classes in 64-channel format")
        
        # Find detections above threshold
        for y in range(height):
            for x in range(width):
                obj_score = objectness[y, x]
                if obj_score > confidence_threshold:
                    # Get class probabilities for this location
                    class_scores = class_probs[:, y, x]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    # Combined confidence
                    confidence = obj_score * class_conf
                    
                    if confidence > confidence_threshold:
                        # Convert from grid coordinates to image coordinates
                        # YOLO outputs are normalized (0-1)
                        center_x = (x + bbox_x[y, x]) / width
                        center_y = (y + bbox_y[y, x]) / height
                        box_w = bbox_w[y, x] / width
                        box_h = bbox_h[y, x] / height
                        
                        # Convert to corner coordinates
                        xmin = center_x - box_w / 2
                        ymin = center_y - box_h / 2
                        xmax = center_x + box_w / 2
                        ymax = center_y + box_h / 2
                        
                        detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
    
    return detections

def filter_detections_by_object_type(detections, object_name):
    """
    Filter detections based on the requested object type.
    
    Args:
        detections: List of all detections
        object_name: The requested object type (e.g., 'vehicle', 'human', 'car')
    
    Returns:
        Filtered list of detections matching the object type
    """
    # COCO class mapping
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Define object type mappings to COCO class IDs
    object_type_mappings = {
        'vehicle': [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
        'car': [2],                  # car
        'truck': [7],               # truck
        'bus': [5],                  # bus
        'motorcycle': [3],           # motorcycle
        'bicycle': [1],              # bicycle
        'human': [0],                # person
        'person': [0],               # person
        'people': [0],               # person
        'animal': [15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
        'dog': [16],                 # dog
        'cat': [15],                 # cat
    }
    
    # Get target class IDs for the requested object type
    if object_name.lower() in object_type_mappings:
        target_class_ids = object_type_mappings[object_name.lower()]
        print(f"🎯 Filtering for '{object_name}' -> COCO classes: {[coco_classes[i] for i in target_class_ids]}")
    else:
        # If not found in mappings, return all detections (backward compatibility)
        print(f"⚠️ No specific mapping for '{object_name}', returning all detections")
        return detections
    
    # Filter detections
    filtered_detections = []
    for detection in detections:
        if detection['class_id'] in target_class_ids:
            filtered_detections.append(detection)
    
    return filtered_detections

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
        
        # Set a confidence threshold (lowered for better detection)
        confidence_threshold = 0.1
        all_detections = []
        
        # Parse model output based on format
        print(f"🔍 Model output shape: {model_output.shape}")
        
        if len(model_output.shape) == 2 and model_output.shape[0] == 84:  # Standard YOLOv8 format (84, 8400)
            # This is standard YOLOv8 output format
            print(f"🔍 Detected standard YOLOv8 format, post-processing...")
            all_detections = post_process_standard_yolo(model_output, confidence_threshold)
        elif len(model_output.shape) in [3, 4]:  # YOLO feature map format
            # This is YOLO raw output - need to post-process
            print(f"🔍 Detected YOLO feature map format, post-processing...")
            all_detections = post_process_yolo_output(model_output, confidence_threshold)
        elif len(model_output.shape) == 2:  # (num_detections, 7) format
            for detection in model_output:
                if len(detection) >= 7:
                    image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
                    if confidence > confidence_threshold:
                        all_detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
        elif len(model_output.shape) == 3:  # (1, num_detections, 7) format
            for detection in model_output[0]:
                if len(detection) >= 7:
                    image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
                    if confidence > confidence_threshold:
                        all_detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
        
        print(f"🔍 Found {len(all_detections)} total detections")
        
        # Filter detections by requested object type
        filtered_detections = filter_detections_by_object_type(all_detections, object_name)
        
        # Print filtered results
        print(f"🎯 Filtered to {len(filtered_detections)} {object_name} detections:")
        for det in filtered_detections:
            print(f"  Class: {det['class_id']}, Confidence: {det['confidence']:.3f}, BBox: {det['bbox']}")
        
        return filtered_detections
        
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