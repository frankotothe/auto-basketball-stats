from collections import defaultdict
import h5py
import os
import cv2
import numpy as np
import torch
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults
from ultralytics import YOLO
import argparse
import json

# Constants
MAX_FRAMES_TO_CHECK = 400
CONFIDENCE_THRESHOLD = 0.6
LOST_FRAMES_THRESHOLD = 10
BALL_CONFIDENCE_THRESHOLD = 0.40  # Still using for the specialized model
PLAYER_CONFIDENCE_THRESHOLD = 0.65
CENTROID_DISTANCE_THRESHOLD = 30  # Maximum distance in pixels between centroids
MASK_IOU_THRESHOLD = 0.5  # Intersection over Union threshold for mask overlap
OVERLAP_FRAMES_THRESHOLD = 2  # Number of consecutive frames before removing overlapping players
RIM_CONFIDENCE_THRESHOLD = 0.5
RIM_REINTRODUCTION_FRAMES = 4

class YOLODetectionStore:
    def __init__(self):
        self.detections = {}  # frame_idx -> detections dict
        
    def add_frame_detections(self, frame_idx, players, ball, rim):
        self.detections[frame_idx] = {
            'players': players,
            'ball': ball,
            'rim': rim
        }
    
    def get_frame_detections(self, frame_idx):
        if frame_idx not in self.detections:
            return [], None, None
        dets = self.detections[frame_idx]
        return dets['players'], dets['ball'], dets['rim']

    # Add a method to update ball detections
    def update_ball_detection(self, frame_idx, ball):
        if frame_idx in self.detections:
            self.detections[frame_idx]['ball'] = ball
        else:
            self.detections[frame_idx] = {
                'players': [],
                'ball': ball,
                'rim': None
            }

class BallDetectionStore:
    def __init__(self):
        self.detections = {}  # frame_idx -> ball detection
        
    def add_ball_detection(self, frame_idx, ball):
        self.detections[frame_idx] = ball
    
    def get_ball_detection(self, frame_idx):
        if frame_idx not in self.detections:
            return None
        return self.detections[frame_idx]

def find_connected_components(mask):
    """Find connected components in a binary mask"""
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    return num_labels, labels

def get_component_areas(labels, num_labels):
    """Get areas of all components"""
    areas = []
    for i in range(1, num_labels):  # Skip background (0)
        area = np.sum(labels == i)
        areas.append((i, area))
    return sorted(areas, key=lambda x: x[1], reverse=True)

class ObjectTracker:
    def __init__(self, obj_key, unique_id):
        self.obj_key = obj_key
        self.unique_id = unique_id
        self.consecutive_low_scores = 0
        self.is_active = True
        self.last_centroid = None
        self.last_mask = None
        self.mask_area = 0
        self.invalid_ball_mask_count = 0
        self.frame_width = None
        self.frame_height = None
        # Add position history
        self.centroid_history = []
        self.max_history = 3
        # Add minimum area threshold as percentage of total mask area
        self.min_component_area_ratio = 0.15  # Components smaller than 20% of largest will be removed
        # Add tracking for frames without high confidence YOLO detection
        self.frames_without_detection = 0
        self.yolo_detection_threshold = 0.25  # 25% accuracy threshold

    def update_score(self, score):
        """Update tracker status based on score"""
        if score < 0:
            self.consecutive_low_scores += 1
        else:
            self.consecutive_low_scores = 0
        
        if self.consecutive_low_scores >= LOST_FRAMES_THRESHOLD:
            self.is_active = False

    def update_mask_and_centroid(self, mask):
        """Update mask, centroid, and area information with split mask handling"""
        if mask is not None and mask.any():
            # Store frame dimensions if not set
            if self.frame_width is None:
                self.frame_height, self.frame_width = mask.shape

            # Find connected components
            num_labels, labels = find_connected_components(mask)
            
            if num_labels > 1:  # Mask is split
                # Get areas of all components
                component_areas = get_component_areas(labels, num_labels)
                largest_area = component_areas[0][1]
                
                # Create new mask with only significant components
                cleaned_mask = np.zeros_like(mask)
                for label_idx, area in component_areas:
                    # Keep component if it's large enough relative to the largest component
                    if area >= largest_area * self.min_component_area_ratio:
                        cleaned_mask = np.logical_or(cleaned_mask, labels == label_idx)
                    """
                    else:
                        print(f"Removing small component (area: {area}) from {self.obj_key}")
                    """
                # Update mask to cleaned version
                mask = cleaned_mask

            # Calculate centroid from cleaned mask
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0:  # If mask is empty after cleaning
                return False
                
            new_centroid = (np.mean(x_indices), np.mean(y_indices))
            
            # Check for large position changes for players only
            if self.obj_key.startswith("player_") and self.last_centroid is not None:
                # Calculate maximum allowed movement (15% of frame dimensions)
                max_x_movement = self.frame_width * 0.15
                max_y_movement = self.frame_height * 0.15
                
                # Calculate actual movement
                x_movement = abs(new_centroid[0] - self.last_centroid[0])
                y_movement = abs(new_centroid[1] - self.last_centroid[1])
                
                # Return early if movement is too large, without updating centroid or mask
                if x_movement > max_x_movement or y_movement > max_y_movement:
                    print(f"Deactivating {self.obj_key} due to excessive movement: "
                          f"dx={x_movement:.1f}, dy={y_movement:.1f}")
                    self.is_active = False
                    return False

            # Update centroid history before updating current centroid
            if self.last_centroid is not None:
                self.centroid_history.append(self.last_centroid)
                if len(self.centroid_history) > self.max_history:
                    self.centroid_history.pop(0)

            # Only update if we haven't detected a jump
            self.last_centroid = new_centroid
            self.last_mask = mask
            self.mask_area = len(x_indices)

            # Ball-specific validation
            if self.obj_key == "ball":
                height = max(y_indices) - min(y_indices)
                width = max(x_indices) - min(x_indices)
                frame_height = mask.shape[0]
                
                aspect_ratio = max(width / height if height != 0 else float('inf'),
                                 height / width if width != 0 else float('inf'))
                relative_height = height / frame_height
                
                if aspect_ratio > 2.5 or relative_height > 0.1:
                    self.invalid_ball_mask_count += 1
                else:
                    self.invalid_ball_mask_count = 0
                
                if self.invalid_ball_mask_count > 2:
                    print(f"Deactivating ball tracker due to invalid mask dimensions: "
                          f"aspect ratio = {aspect_ratio:.2f}, relative height = {relative_height:.2f}")
                    self.is_active = False
                    return False

            return True
        else:
            self.last_centroid = None
            self.last_mask = None
            self.mask_area = 0
            return False
            
    def increment_no_detection(self):
        """Increment counter for frames without high-confidence YOLO detection"""
        self.frames_without_detection += 1
        
        # Check if we've reached the threshold of 8 frames without detection
        if self.frames_without_detection >= 100:
            print(f"Deactivating {self.obj_key} due to {self.frames_without_detection} frames without YOLO detection above {self.yolo_detection_threshold*100}% confidence")
            self.is_active = False
            return False
        return True
        
    def reset_no_detection(self):
        """Reset counter when a high-confidence detection is found"""
        self.frames_without_detection = 0

class ReintroductionTracker:
    def __init__(self, confidence_threshold):
        self.consecutive_detections = []
        self.required_consecutive_frames = 2
        self.confidence_threshold = confidence_threshold
        # Add tracking of frame numbers for debugging
        self.detection_frames = []

    def add_detection(self, detection, frame_idx):
        # Store both detection and frame number
        self.consecutive_detections.append(detection)
        self.detection_frames.append(frame_idx)
        
        # Print detailed info about the detection
        if detection is None:
            print(f"[BALL DEBUG] Frame {frame_idx}: Added NULL detection to reintroduction queue")
        else:
            conf = detection.get('confidence', 'unknown')
            centroid = detection.get('centroid', 'unknown')
            print(f"[BALL DEBUG] Frame {frame_idx}: Added detection with confidence {conf:.4f} at {centroid}")
        
        # Keep only most recent N frames
        if len(self.consecutive_detections) > self.required_consecutive_frames:
            self.consecutive_detections.pop(0)
            self.detection_frames.pop(0)
            
        # Print current state of consecutive detections
        print(f"[BALL DEBUG] Frame {frame_idx}: Current reintroduction queue has {len(self.consecutive_detections)}/{self.required_consecutive_frames} detections")

    def should_reintroduce(self, frame_idx):
        if len(self.consecutive_detections) < self.required_consecutive_frames:
            print(f"[BALL DEBUG] Frame {frame_idx}: Not enough consecutive detections ({len(self.consecutive_detections)}/{self.required_consecutive_frames})")
            return False, None
        
        valid_detections = []
        for i, (det, det_frame) in enumerate(zip(self.consecutive_detections, self.detection_frames)):
            if det is not None and det.get('confidence', 0) > self.confidence_threshold:
                valid_detections.append((det, det_frame))
                print(f"[BALL DEBUG] Frame {frame_idx}: Valid detection in frame {det_frame} with confidence {det.get('confidence', 0):.4f}")
            elif det is None:
                print(f"[BALL DEBUG] Frame {frame_idx}: Invalid NULL detection in frame {det_frame}")
            else:
                print(f"[BALL DEBUG] Frame {frame_idx}: Low confidence detection in frame {det_frame}: {det.get('confidence', 0):.4f} < {self.confidence_threshold}")
        
        if len(valid_detections) == self.required_consecutive_frames:
            print(f"[BALL DEBUG] Frame {frame_idx}: REINTRODUCTION APPROVED - {len(valid_detections)} consecutive valid detections")
            return True, valid_detections[-1][0]  # Return most recent valid detection
        
        print(f"[BALL DEBUG] Frame {frame_idx}: Not enough valid detections ({len(valid_detections)}/{self.required_consecutive_frames})")
        return False, None


def log_object_tracking(trackers, frame_idx, h5_file):
    """Log tracking data in an efficient HDF5 format"""
    frame_group = h5_file.create_group(f"frame_{frame_idx:05d}")  # Zero-padded frame numbers
    
    for obj_key, tracker in trackers.items():
        if tracker.is_active and tracker.last_mask is not None:
            y_coords, x_coords = np.where(tracker.last_mask)
            if len(x_coords) > 0:
                # Store coordinates in a compressed dataset
                obj_dataset = frame_group.create_dataset(
                    f"{obj_key}",  # e.g., "ball" or "player_1"
                    data=np.column_stack((x_coords, y_coords)),
                    dtype=np.uint16,
                    compression="gzip",
                    compression_opts=9
                )
                
                # Store metadata as attributes
                obj_dataset.attrs['object_id'] = tracker.unique_id
                if tracker.last_centroid is not None:
                    obj_dataset.attrs['centroid_x'] = tracker.last_centroid[0]
                    obj_dataset.attrs['centroid_y'] = tracker.last_centroid[1]

def preprocess_video_with_yolo(video_path, yolo_model, ball_yolo_model, total_frames):
    """Batch process entire video with YOLO"""
    print("Pre-processing video with YOLO...")
    detection_store = YOLODetectionStore()
    ball_detection_store = BallDetectionStore()
    vcap = cv2.VideoCapture(video_path)
    
    # Process in batches of frames
    batch_size = 4
    frames = []
    frame_indices = []
    
    for frame_idx in range(total_frames):
        if frame_idx % 100 == 0:
            print(f"Pre-processing frame {frame_idx}/{total_frames}")
            
        ok_frame, frame = vcap.read()
        if not ok_frame:
            break
            
        frames.append(frame)
        frame_indices.append(frame_idx)
        
        # Process batch when full or at end
        if len(frames) == batch_size or frame_idx == total_frames - 1:
            # Process with main YOLO model for players and rim
            results = yolo_model(frames, verbose=False)
            
            # Process with specialized ball YOLO model
            ball_results = ball_yolo_model(frames, verbose=False)
            
            # Process each frame's results
            for batch_idx, (result, ball_result) in enumerate(zip(results, ball_results)):
                curr_frame_idx = frame_indices[batch_idx]
                players = []
                rim = None
                ball = None
                
                # Process main YOLO results (players and rim only)
                for box in result.boxes:
                    class_name = result.names[box.cls[0].item()].lower()
                    x, y, w, h = box.xywh[0].tolist()
                    confidence = box.conf[0].item()
                    
                    # Calculate normalized coordinates
                    norm_x = x / frame.shape[1]
                    norm_y = y / frame.shape[0]
                    norm_x1 = (x - w/2) / frame.shape[1]
                    norm_y1 = (y - h/2) / frame.shape[0]
                    norm_x2 = (x + w/2) / frame.shape[1]
                    norm_y2 = (y + h/2) / frame.shape[0]
                    
                    detection_data = {
                        'confidence': confidence,
                        'box_tlbr_norm_list': [[(norm_x1, norm_y1), (norm_x2, norm_y2)]],
                        'fg_xy_norm_list': [(norm_x, norm_y)],
                        'bg_xy_norm_list': [],
                        'original_box': (int(x - w/2), int(y - h/2), int(w), int(h)),
                        'centroid': (x, y)
                    }
                    
                    if class_name == 'player' and confidence > PLAYER_CONFIDENCE_THRESHOLD:
                        players.append(detection_data)
                    elif class_name == 'rim' and confidence > RIM_CONFIDENCE_THRESHOLD:
                        rim = detection_data
                    # Ignore ball detections from main model
                
                # Process specialized ball YOLO results
                for box in ball_result.boxes:
                    class_name = ball_result.names[box.cls[0].item()].lower()
                    if class_name == 'ball':
                        x, y, w, h = box.xywh[0].tolist()
                        confidence = box.conf[0].item()
                        
                        # Only process if above threshold
                        if confidence > BALL_CONFIDENCE_THRESHOLD:
                            # Calculate normalized coordinates
                            norm_x = x / frame.shape[1]
                            norm_y = y / frame.shape[0]
                            norm_x1 = (x - w/2) / frame.shape[1]
                            norm_y1 = (y - h/2) / frame.shape[0]
                            norm_x2 = (x + w/2) / frame.shape[1]
                            norm_y2 = (y + h/2) / frame.shape[0]
                            
                            ball = {
                                'confidence': confidence,
                                'box_tlbr_norm_list': [[(norm_x1, norm_y1), (norm_x2, norm_y2)]],
                                'fg_xy_norm_list': [(norm_x, norm_y)],
                                'bg_xy_norm_list': [],
                                'original_box': (int(x - w/2), int(y - h/2), int(w), int(h)),
                                'centroid': (x, y)
                            }
                            # Store in specialized ball detection store
                            ball_detection_store.add_ball_detection(curr_frame_idx, ball)
                            break  # Take the first high-confidence ball detection
                
                # Store in main detection store (with ball from specialized model)
                detection_store.add_frame_detections(curr_frame_idx, players, ball, rim)
            
            # Clear batch
            frames = []
            frame_indices = []
    
    vcap.release()
    return detection_store, ball_detection_store


def calculate_mask_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks"""
    if mask1 is None or mask2 is None:
        return 0.0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def calculate_centroid_distance(centroid1, centroid2):
    """Calculate pixel distance between two centroids"""
    if centroid1 is None or centroid2 is None:
        return float('inf')
    return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

def check_for_overlaps(trackers):
    """Check for overlapping player trackers and keep the one with more consistent trajectory"""
    to_remove = set()
    
    # Convert trackers dict to list of tuples for easier processing
    tracker_items = [(k, v) for k, v in trackers.items() if v.is_active and k != "ball"]
    
    # If we have less than 2 active trackers, no overlaps possible
    if len(tracker_items) < 2:
        return to_remove
    
    for i, (key1, tracker1) in enumerate(tracker_items):
        for key2, tracker2 in tracker_items[i+1:]:
            # Skip if either tracker is already marked for removal
            if key1 in to_remove or key2 in to_remove:
                continue
                
            # Skip if either centroid is None
            if tracker1.last_centroid is None or tracker2.last_centroid is None:
                continue
                
            # Calculate centroid distance and mask IoU
            centroid_dist = calculate_centroid_distance(tracker1.last_centroid, tracker2.last_centroid)
            mask_iou = calculate_mask_iou(tracker1.last_mask, tracker2.last_mask)
            
            # Check if trackers are overlapping
            if (centroid_dist < CENTROID_DISTANCE_THRESHOLD and mask_iou > MASK_IOU_THRESHOLD):
                # Calculate trajectory consistency
                consistency1 = calculate_trajectory_consistency(tracker1)
                consistency2 = calculate_trajectory_consistency(tracker2)
                
                # Remove the tracker with less consistent trajectory
                if consistency1 < consistency2:
                    print(f"Removing {key1} due to less consistent trajectory in overlap")
                    to_remove.add(key1)
                else:
                    print(f"Removing {key2} due to less consistent trajectory in overlap")
                    to_remove.add(key2)
    
    return to_remove

def calculate_trajectory_consistency(tracker):
    """Calculate how consistent a tracker's movement has been"""
    if len(tracker.centroid_history) < 2:
        return 0.0
    
    # Calculate average movement vector from history
    movements = []
    for i in range(len(tracker.centroid_history) - 1):
        dx = tracker.centroid_history[i+1][0] - tracker.centroid_history[i][0]
        dy = tracker.centroid_history[i+1][1] - tracker.centroid_history[i][1]
        movements.append((dx, dy))
    
    # Calculate variance in movement
    if not movements:
        return 0.0
    
    avg_dx = sum(m[0] for m in movements) / len(movements)
    avg_dy = sum(m[1] for m in movements) / len(movements)
    
    variance = sum((m[0] - avg_dx)**2 + (m[1] - avg_dy)**2 for m in movements)
    
    # Lower variance means more consistent movement
    return 1.0 / (1.0 + variance)

def process_reintroductions(missing_player_slots, detected_players, trackers, unique_id_counter):
    """Process reintroductions ensuring one-to-one mapping between detections and slots"""
    reintroductions = {}
    
    # Filter out detections that overlap with existing trackers
    valid_detections = [
        p for p in detected_players 
        if not is_detection_overlapping(p, trackers)
    ]
    
    # Map detections to empty slots one-to-one
    for slot, detection in zip(missing_player_slots, valid_detections):
        reintroductions[slot] = {
            'detection': detection,
            'unique_id': unique_id_counter + len(reintroductions)
        }
    
    return reintroductions

def get_missing_player_slots(trackers):
    """Get list of empty player slots"""
    expected_players = set(f"player_{i}" for i in range(10))
    active_players = set(k for k, v in trackers.items() 
                        if k.startswith("player_") and v.is_active)
    return sorted(list(expected_players - active_players))


def is_detection_overlapping(detection, active_trackers):
    """Check if a new detection overlaps with any existing tracked players"""
    if 'centroid' not in detection:
        return False
        
    detection_centroid = detection['centroid']
    
    for tracker in active_trackers.values():
        if not tracker.is_active or tracker.last_centroid is None:
            continue
            
        distance = calculate_centroid_distance(detection_centroid, tracker.last_centroid)
        if distance < CENTROID_DISTANCE_THRESHOLD:
            return True
                
    return False

def find_objects_in_frame(frame_idx, detection_store):
    """Get pre-computed detections for frame"""
    return detection_store.get_frame_detections(frame_idx)

def find_initialization_frame(detection_store):
    """Find frame with required players and ball"""
    print("Starting initialization frame search...")
    
    for frame_idx in range(MAX_FRAMES_TO_CHECK):
        players, ball, _ = detection_store.get_frame_detections(frame_idx)
        print(f"Frame {frame_idx}: Found {len(players)} players and {1 if ball else 0} ball")
        
        if len(players) >= 4 and ball is not None:
            print(f"\nFound suitable frame at index {frame_idx}")
            return frame_idx, players, ball
    
    raise Exception(f"Could not find suitable initialization frame in first {MAX_FRAMES_TO_CHECK} frames")


def encode_mask_rle(mask):
    """
    Convert a binary mask to Run Length Encoding (RLE).
    Returns: List of (start_pos, run_length) tuples and mask shape
    """
    # Flatten mask
    mask_flat = mask.flatten()
    # Find positions where values change
    diff = np.diff(np.concatenate(([0], mask_flat, [0])))
    runs_start = np.where(diff == 1)[0]
    runs_end = np.where(diff == -1)[0]
    
    # Create RLE encoding
    run_lengths = runs_end - runs_start
    rle = list(zip(runs_start, run_lengths))
    return rle, mask.shape


def compress_mask(mask, scale_factor=4):
    """
    Downsample mask by given scale factor using max pooling
    """
    h, w = mask.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    mask_reshaped = mask.reshape(new_h, scale_factor, new_w, scale_factor)
    return mask_reshaped.any(axis=(1,3))


# Replace your log_object_tracking function with this version
def log_object_tracking(trackers, frame_idx, h5_file):
    """Log tracking data using RLE compression"""
    frame_group = h5_file.create_group(f"frame_{frame_idx:05d}")
    
    for obj_key, tracker in trackers.items():
        if tracker.is_active and tracker.last_mask is not None:
            # Create object group
            obj_group = frame_group.create_group(obj_key)
            
            # Store downsampled RLE mask
            downsampled_mask = compress_mask(tracker.last_mask, scale_factor=4)
            rle, shape = encode_mask_rle(downsampled_mask)
            
            # Store RLE data efficiently
            if rle:  # Only store if mask is not empty
                starts, lengths = zip(*rle)
                rle_data = np.array(list(zip(starts, lengths)), dtype=np.uint32)
                obj_group.create_dataset('mask_rle', data=rle_data, 
                                       compression="gzip", compression_opts=9)
                obj_group.create_dataset('mask_shape', data=np.array(shape))
            
            # Store metadata
            obj_group.attrs['object_id'] = tracker.unique_id
            if tracker.last_centroid is not None:
                obj_group.attrs['centroid_x'] = tracker.last_centroid[0]
                obj_group.attrs['centroid_y'] = tracker.last_centroid[1]


def main():
    # Define pathing & device usage
    video_path = "../TestClip2.mp4"
    model_path = "model_weights/large_custom_sam2.pt"
    yolo_model_path = "model_weights/v11.pt"
    ball_yolo_model_path = "model_weights/BestBallYolo.pt"  # New specialized ball model
    unique_id_start = 0  # Initialize here, before config handling

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file for segment processing")
    args = parser.parse_args()
    
    # Updated output paths for separate videos
    output_original = "output_original.mp4"
    output_masks = "output_masks.mp4"
    output_labels = "output_labels.mp4"
    tracking_h5_path = "tracking_logs/object_tracking.h5"
    tracking_data = {}
    MASK_DOWNSAMPLE_FACTOR = 4
    device, dtype = "cpu", torch.float32
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16

    # Override with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            video_path = config.get("video_path", video_path)
            output_original = config.get("output_original", output_original)
            output_masks = config.get("output_masks", output_masks)
            output_labels = config.get("output_labels", output_labels)
            tracking_h5_path = config.get("tracking_h5_path", tracking_h5_path)
            unique_id_start = config.get("unique_id_start", unique_id_start)
            # Could also add ball_yolo_model_path to config if needed
    
    # Load models
    print("Loading models...")
    yolo_model = YOLO(yolo_model_path)
    ball_yolo_model = YOLO(ball_yolo_model_path)  # Load specialized ball model
    model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
    sammodel.to(device=device, dtype=dtype)
    
    # Get video properties
    vcap = cv2.VideoCapture(video_path)
    frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vcap.release()
    
    # Pre-process entire video with YOLO (now using both models)
    detection_store, _ = preprocess_video_with_yolo(
        video_path, 
        yolo_model, 
        ball_yolo_model,  # Pass specialized ball model
        total_frames
    )
    
    # Find initialization frame using pre-computed detections
    init_frame_idx, player_points, ball_point = find_initialization_frame(detection_store)
    print(f"Found initialization frame at index {init_frame_idx}")
    
    # Create prompts from detections
    unique_id_counter = unique_id_start
    prompts_per_frame_index[init_frame_idx] = {
        f"player_{i}": point_data
        for i, point_data in enumerate(player_points[:10])  # Limit to max 10 players (0-9)
    }
    prompts_per_frame_index[init_frame_idx]["ball"] = ball_point
    
    # Set up object trackers
    trackers = {}
    for obj_key in prompts_per_frame_index[init_frame_idx].keys():
        trackers[obj_key] = ObjectTracker(obj_key, unique_id_counter)
        unique_id_counter += 1
    
    # Set up memory storage
    memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)
    
    # Set up reintroduction trackers
    ball_reintroduction = ReintroductionTracker(BALL_CONFIDENCE_THRESHOLD)
    rim_reintroduction = ReintroductionTracker(RIM_CONFIDENCE_THRESHOLD)
    rim_reintroduction.required_consecutive_frames = RIM_REINTRODUCTION_FRAMES
    
    # Process video
    vcap = cv2.VideoCapture(video_path)
    
    # Create separate VideoWriter objects for each output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_original = cv2.VideoWriter(output_original, fourcc, fps, (frame_width, frame_height))
    out_masks = cv2.VideoWriter(output_masks, fourcc, fps, (frame_width, frame_height))
    out_labels = cv2.VideoWriter(output_labels, fourcc, fps, (frame_width, frame_height))
    
    try:
        os.makedirs(os.path.dirname(tracking_h5_path), exist_ok=True)
        with h5py.File(tracking_h5_path, 'w') as h5_file:
            h5_file.attrs['mask_downsample_factor'] = MASK_DOWNSAMPLE_FACTOR
            h5_file.attrs['video_width'] = frame_width
            h5_file.attrs['video_height'] = frame_height
            h5_file.attrs['fps'] = fps
            
            for frame_idx in range(total_frames):
                if frame_idx % 100 == 0:
                    print(f"Processing frame {frame_idx}/{total_frames}")
                    
                ok_frame, frame = vcap.read()
                if not ok_frame:
                    break
                
                # Get pre-computed detections for current frame
                detected_players, detected_ball, detected_rim = find_objects_in_frame(frame_idx, detection_store)
                
                # Update YOLO detection tracking for existing trackers
                for obj_key, tracker in list(trackers.items()):
                    if not tracker.is_active:
                        continue
                        
                    # Check if this object has a high-confidence detection in this frame
                    high_confidence_detection = False
                    
                    # For players
                    if obj_key.startswith("player_") and detected_players:
                        # Find if this player has a matching detection
                        if tracker.last_centroid is not None:
                            for player_detection in detected_players:
                                if 'centroid' in player_detection and 'confidence' in player_detection:
                                    dist = calculate_centroid_distance(tracker.last_centroid, player_detection['centroid'])
                                    if dist < CENTROID_DISTANCE_THRESHOLD and player_detection['confidence'] > tracker.yolo_detection_threshold:
                                        high_confidence_detection = True
                                        break
                    
                    # For ball
                    elif obj_key == "ball" and detected_ball:
                        if tracker.last_centroid is not None and 'centroid' in detected_ball and 'confidence' in detected_ball:
                            dist = calculate_centroid_distance(tracker.last_centroid, detected_ball['centroid'])
                            if dist < CENTROID_DISTANCE_THRESHOLD and detected_ball['confidence'] > tracker.yolo_detection_threshold:
                                high_confidence_detection = True
                                print(f"[BALL DEBUG] Frame {frame_idx}: Matched YOLO ball detection with current tracker (distance: {dist:.2f}px)")
                            else:
                                print(f"[BALL DEBUG] Frame {frame_idx}: YOLO ball detection does NOT match current tracker")
                                if dist >= CENTROID_DISTANCE_THRESHOLD:
                                    print(f"[BALL DEBUG] Frame {frame_idx}: Distance too large: {dist:.2f}px >= {CENTROID_DISTANCE_THRESHOLD}px")
                                if detected_ball['confidence'] <= tracker.yolo_detection_threshold:
                                    print(f"[BALL DEBUG] Frame {frame_idx}: Confidence too low: {detected_ball['confidence']:.4f} <= {tracker.yolo_detection_threshold}")
                    
                    # For rim
                    elif obj_key == "rim" and detected_rim:
                        if tracker.last_centroid is not None and 'centroid' in detected_rim and 'confidence' in detected_rim:
                            dist = calculate_centroid_distance(tracker.last_centroid, detected_rim['centroid'])
                            if dist < CENTROID_DISTANCE_THRESHOLD and detected_rim['confidence'] > tracker.yolo_detection_threshold:
                                high_confidence_detection = True
                    
                    # Update the no-detection counter or reset it
                    if high_confidence_detection:
                        if obj_key == "ball":
                            print(f"[BALL DEBUG] Frame {frame_idx}: Ball has high-confidence YOLO match, resetting no-detection counter")
                        tracker.reset_no_detection()
                    else:
                        if obj_key == "ball":
                            print(f"[BALL DEBUG] Frame {frame_idx}: Ball has NO high-confidence YOLO match, incrementing no-detection counter")
                        tracker.increment_no_detection()
                
                # Handle rim detection and tracking
                if "rim" not in trackers or not trackers["rim"].is_active:
                    rim_reintroduction.add_detection(detected_rim, frame_idx)
                    should_reintroduce, rim_data = rim_reintroduction.should_reintroduce(frame_idx)
                    
                    if should_reintroduce:
                        print(f"Introducing rim tracking at frame {frame_idx}")
                        trackers["rim"] = ObjectTracker("rim", unique_id_counter)
                        unique_id_counter += 1
                        if frame_idx not in prompts_per_frame_index:
                            prompts_per_frame_index[frame_idx] = {}
                        prompts_per_frame_index[frame_idx]["rim"] = rim_data
                
                # Check for lost ball and handle reintroduction
                if "ball" not in trackers or not trackers["ball"].is_active:
                    # Print reason for missing ball if it was previously active
                    if "ball" in trackers:
                        print(f"[BALL DEBUG] Frame {frame_idx}: Ball tracking is INACTIVE")
                        # Print more details if ball was deactivated recently
                        if trackers["ball"].consecutive_low_scores >= LOST_FRAMES_THRESHOLD:
                            print(f"[BALL DEBUG] Frame {frame_idx}: Ball was deactivated due to {trackers['ball'].consecutive_low_scores} consecutive low scores")
                        if hasattr(trackers["ball"], 'invalid_ball_mask_count') and trackers["ball"].invalid_ball_mask_count > 2:
                            print(f"[BALL DEBUG] Frame {frame_idx}: Ball was deactivated due to invalid mask dimensions")
                        if trackers["ball"].frames_without_detection >= 12:
                            print(f"[BALL DEBUG] Frame {frame_idx}: Ball was deactivated due to {trackers['ball'].frames_without_detection} frames without YOLO detection")
                    else:
                        print(f"[BALL DEBUG] Frame {frame_idx}: No ball tracker exists yet")
                    
                    # Log detailed information about the current ball detection
                    if detected_ball:
                        print(f"[BALL DEBUG] Frame {frame_idx}: YOLO detected a ball with confidence {detected_ball['confidence']:.4f} at {detected_ball['centroid']}")
                    else:
                        print(f"[BALL DEBUG] Frame {frame_idx}: YOLO did NOT detect any ball in this frame")
                    
                    # Add to reintroduction tracker with modified function signature
                    ball_reintroduction.add_detection(detected_ball, frame_idx)
                    
                    # Check if reintroduction conditions are met
                    should_reintroduce, ball_data = ball_reintroduction.should_reintroduce(frame_idx)
                    
                    if should_reintroduce:
                        print(f"[BALL DEBUG] Frame {frame_idx}: REINTRODUCING BALL with confidence {ball_data['confidence']:.4f} at {ball_data['centroid']}")
                        trackers["ball"] = ObjectTracker("ball", unique_id_counter)
                        unique_id_counter += 1
                        if frame_idx not in prompts_per_frame_index:
                            prompts_per_frame_index[frame_idx] = {}
                        prompts_per_frame_index[frame_idx]["ball"] = ball_data
                        ball_reintroduction = ReintroductionTracker(BALL_CONFIDENCE_THRESHOLD)
                    else:
                        print(f"[BALL DEBUG] Frame {frame_idx}: Ball reintroduction conditions NOT met")
                
                # Handle player reintroductions
                missing_slots = get_missing_player_slots(trackers)
                if missing_slots and detected_players:
                    reintroductions = process_reintroductions(
                        missing_slots,
                        detected_players,
                        trackers,
                        unique_id_counter
                    )
                    
                    for slot, data in reintroductions.items():
                        print(f"Reintroducing {slot} at frame {frame_idx}")
                        trackers[slot] = ObjectTracker(slot, data['unique_id'])
                        if frame_idx not in prompts_per_frame_index:
                            prompts_per_frame_index[frame_idx] = {}
                        prompts_per_frame_index[frame_idx][slot] = data['detection']
                        unique_id_counter = max(unique_id_counter, data['unique_id'] + 1)
                
                # Encode frame data
                encoded_imgs_list, _, _ = sammodel.encode_image(
                    frame,
                    max_side_length=1024,
                    use_square_sizing=True
                )
                
                # Handle initialization frame
                if frame_idx in prompts_per_frame_index:
                    for obj_key, obj_prompts in prompts_per_frame_index[frame_idx].items():
                        print(f"Initializing tracking for {obj_key}")
                        init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(
                            encoded_imgs_list, 
                            box_tlbr_norm_list=obj_prompts['box_tlbr_norm_list'],
                            fg_xy_norm_list=obj_prompts['fg_xy_norm_list'],
                            bg_xy_norm_list=obj_prompts['bg_xy_norm_list']
                        )
                        memory_per_obj_dict[obj_key].store_prompt_result(frame_idx, init_mem, init_ptr)
                
                # Track objects
                combined_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
                player_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
                ball_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
                rim_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
                
                # First pass: validate movements and generate masks
                active_objects = []
                for obj_key, tracker in trackers.items():
                    if not tracker.is_active:
                        continue
                        
                    obj_memory = memory_per_obj_dict[obj_key]
                    obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                        encoded_imgs_list, **obj_memory.to_dict()
                    )
                    
                    # Update tracker status based on score
                    score_value = obj_score.item()
                    if obj_key == "ball":
                        print(f"[BALL DEBUG] Frame {frame_idx}: Ball SAM tracking score: {score_value:.4f}")
                    
                    tracker.update_score(score_value)
                    if not tracker.is_active:
                        if obj_key == "ball":
                            print(f"[BALL DEBUG] Frame {frame_idx}: Removing ball due to consecutive low scores")
                        else:
                            print(f"Removing {obj_key} due to consecutive low scores")
                        continue
                    
                    # Create mask for movement validation
                    obj_mask = torch.nn.functional.interpolate(
                        mask_preds[:, best_mask_idx, :, :],
                        size=combined_mask_result.shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
                    
                    # Validate movement first - if invalid, skip all further processing
                    mask_valid = tracker.update_mask_and_centroid(obj_mask_binary)
                    if not mask_valid:
                        if obj_key == "ball":
                            print(f"[BALL DEBUG] Frame {frame_idx}: Ball mask validation FAILED")
                        continue
                    
                    if obj_key == "ball":
                        print(f"[BALL DEBUG] Frame {frame_idx}: Ball mask validation PASSED")
                    
                    # Only store memory and continue processing if movement was valid
                    obj_memory.store_result(frame_idx, mem_enc, obj_ptr)
                    active_objects.append(obj_key)
                    
                    # Update visualization masks
                    if obj_key == "ball":
                        ball_mask_result = np.bitwise_or(ball_mask_result, obj_mask_binary)
                    elif obj_key == "rim":
                        rim_mask_result = np.bitwise_or(rim_mask_result, obj_mask_binary)
                    else:
                        player_mask_result = np.bitwise_or(player_mask_result, obj_mask_binary)
                    combined_mask_result = np.bitwise_or(combined_mask_result, obj_mask_binary)
                
                # Now check for overlaps using trajectory consistency
                if frame_idx > 0:  # Skip first frame to allow initialization
                    to_remove = check_for_overlaps(trackers)
                    if to_remove:
                        for obj_key in to_remove:
                            if trackers[obj_key].is_active:
                                print(f"Removing {obj_key} due to less consistent trajectory in overlap")
                                trackers[obj_key].is_active = False
                                if obj_key in active_objects:
                                    active_objects.remove(obj_key)
                
                # After tracking objects, add debug information about ball masks if it exists
                if "ball" in trackers and trackers["ball"].is_active:
                    tracker = trackers["ball"]
                    if tracker.last_mask is not None:
                        y_indices, x_indices = np.where(tracker.last_mask)
                        if len(x_indices) > 0:
                            height = max(y_indices) - min(y_indices)
                            width = max(x_indices) - min(x_indices)
                            frame_height = tracker.last_mask.shape[0]
                            
                            aspect_ratio = max(width / height if height != 0 else float('inf'),
                                            height / width if width != 0 else float('inf'))
                            relative_height = height / frame_height
                            
                            print(f"[BALL DEBUG] Frame {frame_idx}: Current ball mask - width={width}, height={height}, aspect_ratio={aspect_ratio:.2f}, rel_height={relative_height:.5f}")
                
                # Log object tracking information
                log_object_tracking(trackers, frame_idx, h5_file)
                
                # Create visualization frame
                vis_mask = np.zeros((*frame.shape[0:2], 3), dtype=np.uint8)
                
                # Layer the masks in specific order: rim -> players -> ball
                vis_mask[rim_mask_result] = [0, 255, 0]         # Green for rim
                vis_mask[player_mask_result] = [255, 255, 255]  # White for players
                vis_mask[ball_mask_result] = [0, 165, 255]      # Orange for ball
                
                # Create info frame with object IDs and tracking details
                info_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                
                # Add tracking labels
                for obj_key, tracker in trackers.items():
                    if tracker.is_active and tracker.last_centroid is not None:
                        label = f"{obj_key}-{tracker.unique_id}"
                        cv2.putText(info_frame, 
                                label, 
                                (int(tracker.last_centroid[0]), int(tracker.last_centroid[1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add status information to info frame
                active_players = len([k for k in active_objects if k.startswith("player_")])
                cv2.putText(info_frame, f"Frame: {frame_idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Active Players: {active_players}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Ball Status: {'Active' if 'ball' in active_objects else 'Lost'}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Rim Status: {'Active' if 'rim' in active_objects else 'Lost'}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frames to respective output videos
                out_original.write(frame)
                out_masks.write(vis_mask)
                out_labels.write(info_frame)
                    
    finally:
        vcap.release()
        out_original.release()
        out_masks.release()
        out_labels.release()
        print(f"Original video saved to: {output_original}")
        print(f"Mask video saved to: {output_masks}")
        print(f"Labels video saved to: {output_labels}")
        print(f"Tracking data saved to: {tracking_h5_path}")


if __name__ == "__main__":
    main()