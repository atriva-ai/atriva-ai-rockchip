from fastapi import APIRouter, UploadFile, File
from app.services import run_object_detection
from app.models import model_manager

router = APIRouter()

@router.get("/models")
async def list_available_models():
    """Returns a list of all available models."""
    return {"available_models": model_manager.list_models()}

@router.post("/inference/detection")
def detect_objects(object_name: str, image: UploadFile = File(...)):
    image_bytes = image.file.read()
    detections = run_object_detection(image_bytes, object_name)
    return {"objects": detections}
