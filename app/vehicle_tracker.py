import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

@dataclass
class Track:
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    age: int
    hits: int
    time_since_update: float
    kalman_filter: Optional[object] = None

class ByteTracker:
    """ByteTrack algorithm implementation for vehicle tracking"""
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.frame_id = 0
        self.next_id = 1
        
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections"""
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_conf_dets = [d for d in detections if d.confidence < self.track_thresh]
        
        # Step 1: Associate high confidence detections with existing tracks
        track_pool = self.tracked_tracks + self.lost_tracks
        matches, unmatched_tracks, unmatched_detections = self._associate_detections_to_tracks(
            high_conf_dets, track_pool, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            detection = high_conf_dets[det_idx]
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.age += 1
            track.hits += 1
            track.time_since_update = 0
            
            # Move to tracked tracks if not already there
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)
                self.tracked_tracks.append(track)
        
        # Step 2: Initialize new tracks from unmatched high confidence detections
        for det_idx in unmatched_detections:
            detection = high_conf_dets[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                age=1,
                hits=1,
                time_since_update=0
            )
            self.tracked_tracks.append(new_track)
            self.next_id += 1
        
        # Step 3: Associate low confidence detections with unmatched tracks
        if low_conf_dets and unmatched_tracks:
            track_pool = [track_pool[i] for i in unmatched_tracks]
            matches_low, _, _ = self._associate_detections_to_tracks(
                low_conf_dets, track_pool, self.match_thresh
            )
            
            for track_idx, det_idx in matches_low:
                track = track_pool[track_idx]
                detection = low_conf_dets[det_idx]
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.age += 1
                track.hits += 1
                track.time_since_update = 0
                
                # Move to tracked tracks if not already there
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                    self.tracked_tracks.append(track)
        
        # Step 4: Update unmatched tracks
        for track_idx in unmatched_tracks:
            track = track_pool[track_idx]
            track.time_since_update += 1
            
            # Move to lost tracks if not already there
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)
                self.lost_tracks.append(track)
        
        # Step 5: Remove old tracks
        self._remove_old_tracks()
        
        return self.tracked_tracks
    
    def _associate_detections_to_tracks(self, detections: List[Detection], tracks: List[Track], 
                                      threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using IoU"""
        if not tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(tracks))), []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Use Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Fallback to greedy assignment if scipy is not available
            track_indices, detection_indices = self._greedy_assignment(iou_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_detections = []
        
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if iou_matrix[track_idx, det_idx] >= threshold:
                matches.append((track_idx, det_idx))
            else:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(det_idx)
        
        # Add unmatched tracks and detections
        for i in range(len(tracks)):
            if i not in [m[0] for m in matches]:
                unmatched_tracks.append(i)
        
        for i in range(len(detections)):
            if i not in [m[1] for m in matches]:
                unmatched_detections.append(i)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _greedy_assignment(self, iou_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """Greedy assignment fallback when scipy is not available"""
        track_indices = []
        detection_indices = []
        
        # Simple greedy assignment
        for i in range(min(iou_matrix.shape)):
            max_iou = np.max(iou_matrix)
            if max_iou > 0:
                track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                track_indices.append(track_idx)
                detection_indices.append(det_idx)
                iou_matrix[track_idx, :] = 0
                iou_matrix[:, det_idx] = 0
        
        return track_indices, detection_indices
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_old_tracks(self):
        """Remove tracks that are too old"""
        current_tracks = []
        for track in self.tracked_tracks:
            if track.time_since_update < self.track_buffer:
                current_tracks.append(track)
            else:
                logger.info(f"Removing old track {track.track_id}")
        self.tracked_tracks = current_tracks
        
        current_lost = []
        for track in self.lost_tracks:
            if track.time_since_update < self.track_buffer:
                current_lost.append(track)
            else:
                logger.info(f"Removing old lost track {track.track_id}")
        self.lost_tracks = current_lost

class VehicleTracker:
    """Main vehicle tracking service using ByteTrack and RKNN models"""
    
    def __init__(self, camera_id: int, config: Optional[Dict] = None):
        self.camera_id = camera_id
        self.config = config or {}
        self.tracker = ByteTracker(
            track_thresh=self.config.get('track_thresh', 0.5),
            track_buffer=self.config.get('track_buffer', 30),
            match_thresh=self.config.get('match_thresh', 0.8)
        )
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }
        
        # Tracking state
        self.is_active = False
        self.frame_count = 0
        self.tracked_vehicles = {}
        
        # Output directories - use shared storage
        self.output_dir = Path(f"/app/shared/vehicle_tracking/camera_{camera_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Vehicle tracker initialized for camera {camera_id}")
    
    def detect_vehicles(self, frame_bytes: bytes) -> List[Detection]:
        """Detect vehicles in the frame using RKNN YOLOv8n model"""
        try:
            from app.services import run_inference, preprocess_image
            
            # Use YOLOv8n for vehicle detection
            model_name = "yolov8n"
            input_shape = (640, 640)  # YOLOv8n input shape
            
            # Preprocess image for model input
            preprocessed_image = preprocess_image(frame_bytes, input_shape)
            
            # Run inference using RKNN
            model_output = run_inference(preprocessed_image, model_name)
            
            # Parse YOLOv8n output format
            detections = []
            confidence_threshold = 0.5
            
            # YOLOv8n output format: [batch, num_detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
            if len(model_output.shape) == 3:
                # Remove batch dimension
                model_output = model_output[0]
            
            for detection in model_output:
                # Get confidence and class probabilities
                bbox = detection[:4]  # x_center, y_center, width, height
                confidence = detection[4]
                class_probs = detection[5:]
                
                if confidence > confidence_threshold:
                    # Find the class with highest probability
                    class_id = np.argmax(class_probs)
                    class_confidence = class_probs[class_id] * confidence
                    
                    if class_id in self.vehicle_classes and class_confidence > confidence_threshold:
                        # Convert center format to x1, y1, x2, y2
                        x_center, y_center, width, height = bbox
                        x1 = max(0, x_center - width / 2)
                        y1 = max(0, y_center - height / 2)
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        detection_obj = Detection(
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=float(class_confidence),
                            class_id=int(class_id),
                            class_name=self.vehicle_classes[class_id]
                        )
                        detections.append(detection_obj)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return []
    
    def track_vehicles(self, frame_bytes: bytes) -> Tuple[bytes, List[Track]]:
        """Track vehicles in the frame and return annotated frame bytes"""
        if not self.is_active:
            return frame_bytes, []
        
        # Detect vehicles
        detections = self.detect_vehicles(frame_bytes)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Annotate frame with tracking results
        annotated_frame_bytes = self._annotate_frame(frame_bytes, tracks)
        
        # Update tracking state
        for track in tracks:
            self.tracked_vehicles[track.track_id] = {
                'bbox': track.bbox,
                'class_name': track.class_name,
                'confidence': track.confidence,
                'age': track.age,
                'last_seen': time.time()
            }
        
        self.frame_count += 1
        return annotated_frame_bytes, tracks
    
    def _annotate_frame(self, frame_bytes: bytes, tracks: List[Track]) -> bytes:
        """Annotate frame with tracking results using PIL instead of OpenCV"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(frame_bytes))
            draw = ImageDraw.Draw(image)
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box (green for tracked vehicles)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                
                # Draw track ID and class
                label = f"{track.class_name}-{track.track_id}"
                # Draw text with black background for better visibility
                bbox = draw.textbbox((x1, y1-25), label, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((x1, y1-25), label, fill=(0, 255, 0), font=font)
            
            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            # Return original frame if annotation fails
            return frame_bytes
    
    def save_annotated_frame(self, frame_bytes: bytes, frame_number: int) -> str:
        """Save annotated frame to output directory and return path"""
        try:
            timestamp = int(time.time())
            filename = f"frame_{frame_number:06d}_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            # Save the annotated frame
            with open(filepath, 'wb') as f:
                f.write(frame_bytes)
            
            logger.info(f"Saved annotated frame: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save annotated frame: {e}")
            return ""
    
    def start_tracking(self):
        """Start vehicle tracking"""
        self.is_active = True
        logger.info(f"Vehicle tracking started for camera {self.camera_id}")
    
    def stop_tracking(self):
        """Stop vehicle tracking"""
        self.is_active = False
        logger.info(f"Vehicle tracking stopped for camera {self.camera_id}")
    
    def get_status(self) -> Dict:
        """Get current tracking status"""
        return {
            'camera_id': self.camera_id,
            'is_active': self.is_active,
            'frame_count': self.frame_count,
            'tracked_vehicles': len(self.tracked_vehicles),
            'output_directory': str(self.output_dir)
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_tracking()
        # Clean up old frames if needed
        try:
            for file in self.output_dir.glob("*.jpg"):
                if time.time() - file.stat().st_mtime > 3600:  # 1 hour
                    file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up old frames: {e}")

# Global tracker instances
vehicle_trackers: Dict[int, VehicleTracker] = {}

def get_vehicle_tracker(camera_id: int, config: Optional[Dict] = None) -> VehicleTracker:
    """Get or create vehicle tracker for a camera"""
    if camera_id not in vehicle_trackers:
        vehicle_trackers[camera_id] = VehicleTracker(camera_id, config)
    return vehicle_trackers[camera_id]

def remove_vehicle_tracker(camera_id: int):
    """Remove vehicle tracker for a camera"""
    if camera_id in vehicle_trackers:
        vehicle_trackers[camera_id].cleanup()
        del vehicle_trackers[camera_id]
        logger.info(f"Vehicle tracker removed for camera {camera_id}")

def process_frame_for_tracking(camera_id: int, frame_bytes: bytes, frame_number: int) -> Tuple[bytes, List[Track], str]:
    """Process a frame for vehicle tracking and return annotated frame bytes, tracks, and saved path"""
    tracker = get_vehicle_tracker(camera_id)
    
    # Track vehicles
    annotated_frame_bytes, tracks = tracker.track_vehicles(frame_bytes)
    
    # Save annotated frame if tracking is active
    saved_path = ""
    if tracker.is_active and tracks:
        saved_path = tracker.save_annotated_frame(annotated_frame_bytes, frame_number)
    
    return annotated_frame_bytes, tracks, saved_path
