from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import run_object_detection, get_available_objects, get_available_models
from app.models import model_manager, ACCELERATORS, MODEL_NAME_MAPPING
from app.shared_data import (
    list_available_cameras, 
    get_frame_info, 
    get_latest_frame, 
    read_frame_file,
    list_camera_frames
)
from fastapi.responses import FileResponse
import os
from threading import Thread
from typing import Optional

router = APIRouter()

ARCHITECTURE = "rk3588"

@router.get("/models")
async def list_available_models():
    """Returns a list of all available models."""
    return {"available_models": model_manager.list_models()}

@router.get("/objects")
async def list_available_objects():
    """Returns a list of all available object types for detection."""
    return {"available_objects": get_available_objects()}

@router.post("/inference/detection")
def detect_objects(object_name: str, image: UploadFile = File(...)):
    """Run object detection on uploaded image."""
    image_bytes = image.file.read()
    detections = run_object_detection(image_bytes, object_name)
    return {"objects": detections}

@router.get("/shared/cameras")
async def list_cameras():
    """List all cameras that have decoded frames available."""
    cameras = list_available_cameras()
    return {"cameras": cameras}

@router.get("/shared/cameras/{camera_id}/frames")
async def get_camera_frames(camera_id: str):
    """Get information about decoded frames for a specific camera."""
    frame_info = get_frame_info(camera_id)
    return frame_info

@router.get("/shared/cameras/{camera_id}/frames/latest")
async def get_camera_latest_frame(camera_id: str):
    """Get the latest decoded frame for a camera."""
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    return FileResponse(latest_frame)

@router.post("/shared/cameras/{camera_id}/inference")
async def detect_objects_in_camera_frame(camera_id: str, object_name: str):
    """Run object detection on the latest frame from a camera."""
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    # Read the frame file
    frame_bytes = read_frame_file(latest_frame)
    if not frame_bytes:
        raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
    
    # Run object detection
    detections = run_object_detection(frame_bytes, object_name)
    
    return {
        "camera_id": camera_id,
        "frame_path": latest_frame,
        "object_name": object_name,
        "detections": detections
    }

@router.get("/shared/cameras/{camera_id}/frames/{frame_index}")
async def get_camera_frame_by_index(camera_id: str, frame_index: int):
    """Get a specific frame by index for a camera."""
    frame_files = list_camera_frames(camera_id)
    if not frame_files:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    if frame_index < 0 or frame_index >= len(frame_files):
        raise HTTPException(
            status_code=400, 
            detail=f"Frame index {frame_index} out of range. Available frames: 0-{len(frame_files)-1}"
        )
    
    return FileResponse(frame_files[frame_index])

# --- Model Info API ---
@router.get("/model/info")
async def get_model_info():
    return {
        "models": list(MODEL_NAME_MAPPING.keys()),
        "objects": get_available_objects(),
        "accelerators": ACCELERATORS,
        "architecture": ARCHITECTURE
    }

# --- Model Load API ---
@router.post("/model/load")
async def load_model(model_name: str):
    """Load a specific model for inference."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    try:
        print(f"🔄 Loading model: {model_name}")
        
        # Load the model using the model manager
        model_path = model_manager.load_model(model_name)
        
        print(f"✅ Model loaded successfully: {model_path}")
        
        return {
            "model_name": model_name,
            "model_path": model_path,
            "architecture": ARCHITECTURE,
            "status": "loaded"
        }
    except Exception as e:
        error_msg = f"❌ Failed to load model {model_name}: {str(e)}"
        print(error_msg)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

# --- Direct Inference API ---
@router.post("/inference/direct")
async def direct_inference(model_name: str, image: UploadFile = File(...)):
    """Run direct inference using a specific model."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    try:
        from app.services import run_inference, preprocess_image
        from config import MODEL_INPUT_SHAPES
        
        image_bytes = image.file.read()
        
        # Get input shape for the model
        input_shape = MODEL_INPUT_SHAPES.get(model_name, (640, 640))
        
        # Preprocess image
        preprocessed_image = preprocess_image(image_bytes, input_shape)
        
        # Run inference
        output = run_inference(preprocessed_image, model_name)
        
        return {
            "model_name": model_name,
            "input_shape": input_shape,
            "output_shape": output.shape if hasattr(output, 'shape') else None,
            "output": output.tolist() if hasattr(output, 'tolist') else str(output)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Inference on Latest Frame API ---
@router.post("/inference/latest-frame")
async def inference_latest_frame(camera_id: str, model_name: str):
    """Run inference on the latest frame using a specific model."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    latest_frame = get_latest_frame(camera_id)
    if not latest_frame:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")
    
    frame_bytes = read_frame_file(latest_frame)
    if not frame_bytes:
        raise HTTPException(status_code=500, detail=f"Failed to read frame file: {latest_frame}")
    
    try:
        from app.services import run_inference, preprocess_image
        from config import MODEL_INPUT_SHAPES
        
        # Get input shape for the model
        input_shape = MODEL_INPUT_SHAPES.get(model_name, (640, 640))
        
        # Preprocess image
        preprocessed_image = preprocess_image(frame_bytes, input_shape)
        
        # Run inference
        output = run_inference(preprocessed_image, model_name)
        
        return {
            "camera_id": camera_id,
            "model_name": model_name,
            "architecture": ARCHITECTURE,
            "frame_path": latest_frame,
            "input_shape": input_shape,
            "output_shape": output.shape if hasattr(output, 'shape') else None,
            "output": output.tolist() if hasattr(output, 'tolist') else str(output)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Background Inference API ---
def background_inference(camera_id: str, model_name: str):
    """Run background inference on all frames for a camera."""
    from app.services import run_inference, preprocess_image
    from config import MODEL_INPUT_SHAPES
    
    try:
        # Get input shape for the model
        input_shape = MODEL_INPUT_SHAPES.get(model_name, (640, 640))
        
        frame_files = list_camera_frames(camera_id)
        results = []
        
        for frame_path in frame_files:
            frame_bytes = read_frame_file(frame_path)
            if not frame_bytes:
                continue
                
            preprocessed_image = preprocess_image(frame_bytes, input_shape)
            output = run_inference(preprocessed_image, model_name)
            
            results.append({
                "frame_path": frame_path,
                "output_shape": output.shape if hasattr(output, 'shape') else None,
                "output": output.tolist() if hasattr(output, 'tolist') else str(output)
            })
        
        print(f"Background inference for camera {camera_id} complete. {len(results)} frames processed.")
        
    except Exception as e:
        print(f"Background inference error: {str(e)}")

@router.post("/inference/background")
async def start_background_inference(camera_id: str, model_name: str):
    """Start background inference on all frames for a camera."""
    if model_name not in MODEL_NAME_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    thread = Thread(target=background_inference, args=(camera_id, model_name), daemon=True)
    thread.start()
    
    return {
        "camera_id": camera_id,
        "model_name": model_name,
        "architecture": ARCHITECTURE,
        "status": "background_inference_started"
    }
