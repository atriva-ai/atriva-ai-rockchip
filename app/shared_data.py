import os
from pathlib import Path
from typing import List, Optional
import glob

# Shared data paths
SHARED_TEMP_PATH = os.getenv("SHARED_TEMP_PATH", "/app/shared")
SHARED_FRAMES_PATH = os.getenv("SHARED_FRAMES_PATH", "/app/frames")

def get_shared_frames_path() -> Path:
    """Get the path to the shared frames directory."""
    return Path(SHARED_FRAMES_PATH)

def get_shared_temp_path() -> Path:
    """Get the path to the shared temp directory."""
    return Path(SHARED_TEMP_PATH)

def list_camera_frames(camera_id: str) -> List[str]:
    """
    List all decoded frames for a specific camera.
    
    Args:
        camera_id: The ID of the camera
        
    Returns:
        List of frame file paths for the camera
    """
    frames_dir = get_shared_frames_path() / camera_id
    if not frames_dir.exists():
        return []
    
    # Find all JPEG files in the camera directory
    frame_files = glob.glob(str(frames_dir / "*.jpg"))
    frame_files.sort()  # Sort by filename for chronological order
    return frame_files

def get_latest_frame(camera_id: str) -> Optional[str]:
    """
    Get the path to the latest decoded frame for a camera.
    
    Args:
        camera_id: The ID of the camera
        
    Returns:
        Path to the latest frame file, or None if no frames exist
    """
    frame_files = list_camera_frames(camera_id)
    if not frame_files:
        return None
    
    return frame_files[-1]  # Return the last (latest) frame

def get_frame_count(camera_id: str) -> int:
    """
    Get the number of decoded frames for a camera.
    
    Args:
        camera_id: The ID of the camera
        
    Returns:
        Number of frames available for the camera
    """
    return len(list_camera_frames(camera_id))

def list_available_cameras() -> List[str]:
    """
    List all camera IDs that have decoded frames available.
    
    Returns:
        List of camera IDs
    """
    frames_dir = get_shared_frames_path()
    if not frames_dir.exists():
        return []
    
    # Get all subdirectories (camera IDs)
    camera_dirs = [d.name for d in frames_dir.iterdir() if d.is_dir()]
    return camera_dirs

def read_frame_file(frame_path: str) -> Optional[bytes]:
    """
    Read a frame file and return its bytes.
    
    Args:
        frame_path: Path to the frame file
        
    Returns:
        Frame file bytes, or None if file doesn't exist
    """
    try:
        with open(frame_path, 'rb') as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return None

def get_frame_info(camera_id: str) -> dict:
    """
    Get information about decoded frames for a camera.
    
    Args:
        camera_id: The ID of the camera
        
    Returns:
        Dictionary with frame information
    """
    frame_count = get_frame_count(camera_id)
    latest_frame = get_latest_frame(camera_id)
    
    return {
        "camera_id": camera_id,
        "frame_count": frame_count,
        "latest_frame": latest_frame,
        "has_frames": frame_count > 0
    } 