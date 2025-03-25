import numpy as np
import cv2
# from openvino import AsyncInferQueue
from openvino.runtime import InferRequest
from app.models import model_manager

def preprocess_image(image_bytes: bytes, input_shape: tuple):
    """Preprocess image to match OpenVINO model input shape."""
    _, _, h, w = input_shape  # Extract height and width from model shape
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

def run_inference(input_data, compiled_model):
    """Runs inference on input data using OpenVINO model."""
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Debug: Check input shape before inference
    print(f"Input shape before inference: {input_data.shape}, expected: {input_layer.shape}")

    # Convert input to OpenVINO-compatible format
    input_tensor = np.array(input_data, dtype=np.float32)

    # Create inference request
    infer_request = compiled_model.create_infer_request()

    # Start inference synchronously
    infer_request.infer({input_layer: input_tensor})
    print("Inference started...")

    # Start inference asynchronously
    # infer_request.start_async({input_layer: input_tensor})

    # Wait for inference to complete
    # infer_request.wait()

    '''
    # Run inference asynchronously
    infer_queue = AsyncInferQueue(compiled_model)

    # Define callback function to handle inference completion
    results = {}

    def callback(request, user_data):
        results["output"] = request.results[output_layer]

    infer_queue.set_callback(callback)
    infer_queue.start_async({input_layer: input_tensor})

    # Debug: Check if the inference is started
    print("Inference started...")

    # Wait until inference is completed
    infer_queue.wait_for_all()
    print("Inference completed.")
    
    return results["output"].tolist()
    '''

    # Get the output
    output = infer_request.get_output_tensor(0).data # # Shape: (1, 1, 200, 7)
    print(f"Output shape after inference: {output.shape}")
    output = np.squeeze(output)  # Remove unnecessary dimensions, now (200, 7)

    return output

def run_object_detection(image_bytes: bytes, object_name: str):

    """Runs inference using a dynamically selected model."""
    compiled_model, input_shape = model_manager.load_model(object_name)  # Load requested model by object name like "face"
    
    # Preprocess image for model input
    image = preprocess_image(image_bytes, input_shape)

    # Run inference
    model_output = run_inference(image, compiled_model)  # Ensure run_inference() is correctly handling input

    # Set a confidence threshold
    confidence_threshold = 0.3
    detections = []

    # Parse model output (example structure, adjust based on actual model)
    for detection in model_output:
        image_id, class_id, confidence, xmin, ymin, xmax, ymax = detection
        if confidence > confidence_threshold:  # Filter detections by confidence
            detections.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
            })

    # Print results
    for det in detections:
        print(f"Class: {det['class_id']}, Confidence: {det['confidence']}, BBox: {det['bbox']}")

    return detections