from collections import defaultdict
import json
import os
import cv2
import numpy as np
import torch
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults
from ultralytics import YOLO

# Constants
MAX_FRAMES_TO_CHECK = 250
CONFIDENCE_THRESHOLD = 0.6
LOST_FRAMES_THRESHOLD = 10
BALL_CONFIDENCE_THRESHOLD = 0.5
PLAYER_CONFIDENCE_THRESHOLD = 0.65
CENTROID_DISTANCE_THRESHOLD = 30  # Maximum distance in pixels between centroids
MASK_IOU_THRESHOLD = 0.5  # Intersection over Union threshold for mask overlap
OVERLAP_FRAMES_THRESHOLD = 2  # Number of consecutive frames before removing overlapping players
RIM_CONFIDENCE_THRESHOLD = 0.5
RIM_REINTRODUCTION_FRAMES = 5

class ObjectTracker:
    def __init__(self, obj_key, unique_id):
        self.obj_key = obj_key
        self.unique_id = unique_id  # Added unique identifier
        self.consecutive_low_scores = 0
        self.is_active = True
        self.last_centroid = None
        self.last_mask = None
        self.mask_area = 0
        self.overlap_count = defaultdict(int)
        self.invalid_ball_mask_count = 0  # New: counter for invalid ball masks

    def update_score(self, score):
        if score < 0:
            self.consecutive_low_scores += 1
        else:
            self.consecutive_low_scores = 0
        
        if self.consecutive_low_scores >= LOST_FRAMES_THRESHOLD:
            self.is_active = False
            
    def update_mask_and_centroid(self, mask):
        """Update mask, centroid, and area information"""
        self.last_mask = mask
        if mask is not None and mask.any():
            y_indices, x_indices = np.where(mask)
            self.last_centroid = (np.mean(x_indices), np.mean(y_indices))
            self.mask_area = len(x_indices)

            # Ball-specific validation
            if self.obj_key == "ball":
                height = max(y_indices) - min(y_indices)
                width = max(x_indices) - min(x_indices)
                frame_height = mask.shape[0]
                
                # Check aspect ratio and relative height
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
        else:
            self.last_centroid = None
            self.mask_area = 0

class ReintroductionTracker:
    def __init__(self, confidence_threshold):
        self.consecutive_detections = []
        self.required_consecutive_frames = 3
        self.confidence_threshold = confidence_threshold

    def add_detection(self, detection):
        self.consecutive_detections.append(detection)
        if len(self.consecutive_detections) > self.required_consecutive_frames:
            self.consecutive_detections.pop(0)

    def should_reintroduce(self):
        if len(self.consecutive_detections) < self.required_consecutive_frames:
            return False, None
            
        valid_detections = [d for d in self.consecutive_detections 
                          if d is not None and d['confidence'] > self.confidence_threshold]
        
        if len(valid_detections) == self.required_consecutive_frames:
            return True, valid_detections[-1]
        return False, None

def log_object_tracking(trackers, frame_idx, json_data):
    """Log object tracking information to a JSON structure"""
    frame_key = f"frame_{frame_idx}"
    json_data[frame_key] = {}
    
    for obj_key, tracker in trackers.items():
        if tracker.is_active and tracker.last_mask is not None:
            # Get coordinates where mask is True
            y_coords, x_coords = np.where(tracker.last_mask)
            if len(x_coords) > 0:  # Only add if mask has True pixels
                json_data[frame_key][f"{obj_key}_{tracker.unique_id}"] = {
                    "type": obj_key,
                    "coords": {
                        "x": x_coords.tolist(),
                        "y": y_coords.tolist()
                    }
                }

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
    """Check for overlapping player trackers using both mask IoU and centroid distance"""
    to_remove = set()
    
    # Convert trackers dict to list of tuples for easier processing
    tracker_items = [(k, v) for k, v in trackers.items() if v.is_active and k != "ball"]
    
    for i, (key1, tracker1) in enumerate(tracker_items):
        for key2, tracker2 in tracker_items[i+1:]:
            # Skip if either tracker is already marked for removal
            if key1 in to_remove or key2 in to_remove:
                continue
                
            # Calculate centroid distance and mask IoU
            centroid_dist = calculate_centroid_distance(tracker1.last_centroid, tracker2.last_centroid)
            mask_iou = calculate_mask_iou(tracker1.last_mask, tracker2.last_mask)
            
            # Check if trackers are overlapping
            if (centroid_dist < CENTROID_DISTANCE_THRESHOLD and mask_iou > MASK_IOU_THRESHOLD):
                tracker1.overlap_count[key2] += 1
                tracker2.overlap_count[key1] += 1
                
                if tracker1.overlap_count[key2] >= OVERLAP_FRAMES_THRESHOLD:
                    # Remove the tracker with smaller mask area (likely partial detection)
                    if tracker1.mask_area <= tracker2.mask_area:
                        to_remove.add(key1)
                    else:
                        to_remove.add(key2)
    
    return to_remove

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

def find_objects_in_frame(frame, yolo_model):
    """Detect players, ball, and rim in a single frame using YOLO"""
    results = yolo_model(frame, verbose=False)
    players = []
    ball = None
    rim = None
    
    for result in results:
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
            elif class_name == 'basketball' and confidence > BALL_CONFIDENCE_THRESHOLD:
                ball = detection_data
            elif class_name == 'rim' and confidence > RIM_CONFIDENCE_THRESHOLD:
                rim = detection_data
                
    return players, ball, rim

def find_initialization_frame(video_path, yolo_model):
    """Find frame with required players and ball"""
    vcap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    print("Starting initialization frame search...")
    try:
        while frame_idx < MAX_FRAMES_TO_CHECK:
            ok_frame, frame = vcap.read()
            if not ok_frame:
                break
                
            players, ball, _ = find_objects_in_frame(frame, yolo_model)  # Ignore rim during initialization
            print(f"Frame {frame_idx}: Found {len(players)} players and {1 if ball else 0} ball")
            
            if len(players) >= 10 and ball is not None:
                print(f"\nFound suitable frame at index {frame_idx}")
                vcap.release()
                return frame_idx, players, ball
                
            frame_idx += 1
        
        vcap.release()
        raise Exception(f"Could not find suitable initialization frame in first {MAX_FRAMES_TO_CHECK} frames")
        
    except Exception as e:
        vcap.release()
        print(f"Initialization failed: {str(e)}")
        raise


def main():
    # Define pathing & device usage
    video_path = "../TestClip.mp4"
    model_path = "model_weights/sam2.1_hiera_large.pt"
    yolo_model_path = "model_weights/v11.pt"
    
    # Updated output paths for separate videos
    output_original = "output_original.mp4"
    output_masks = "output_masks.mp4"
    output_labels = "output_labels.mp4"
    tracking_json_path = "tracking_logs/object_tracking.json"
    tracking_data = {}

    device, dtype = "cpu", torch.float32
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    
    # Load models
    print("Loading models...")
    yolo_model = YOLO(yolo_model_path)
    model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
    sammodel.to(device=device, dtype=dtype)
    
    # Find initialization frame
    print("Searching for initialization frame...")
    init_frame_idx, player_points, ball_point = find_initialization_frame(video_path, yolo_model)
    print(f"Found initialization frame at index {init_frame_idx}")
    
    # Create prompts from detections
    unique_id_counter = 0
    prompts_per_frame_index = {
        init_frame_idx: {
            f"player_{unique_id_counter+i}": point_data
            for i, point_data in enumerate(player_points)
        }
    }
    # Add ball point
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
    player_reintroduction_trackers = {}
    lost_frames_counters = defaultdict(int)
    
    # Process video
    vcap = cv2.VideoCapture(video_path)
    
    # Get video properties for output
    frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    
    # Create separate VideoWriter objects for each output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_original = cv2.VideoWriter(output_original, fourcc, fps, (frame_width, frame_height))
    out_masks = cv2.VideoWriter(output_masks, fourcc, fps, (frame_width, frame_height))
    out_labels = cv2.VideoWriter(output_labels, fourcc, fps, (frame_width, frame_height))
    
    vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    try:
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(total_frames):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")
                
            ok_frame, frame = vcap.read()
            if not ok_frame:
                break
            
            # Get current detections
            detected_players, detected_ball, detected_rim = find_objects_in_frame(frame, yolo_model)
            
            # Handle rim detection and tracking
            if "rim" not in trackers and detected_rim is not None:
                rim_reintroduction.add_detection(detected_rim)
                should_reintroduce, rim_data = rim_reintroduction.should_reintroduce()
                
                if should_reintroduce:
                    print(f"Introducing rim tracking at frame {frame_idx}")
                    trackers["rim"] = ObjectTracker("rim", unique_id_counter)
                    unique_id_counter += 1
                    if frame_idx not in prompts_per_frame_index:
                        prompts_per_frame_index[frame_idx] = {}
                    prompts_per_frame_index[frame_idx]["rim"] = rim_data
            
            # Update lost frames counters and handle reintroductions
            for obj_key, tracker in list(trackers.items()):
                if not tracker.is_active:
                    lost_frames_counters[obj_key] += 1
                else:
                    lost_frames_counters[obj_key] = 0
                
                # Handle reintroduction logic
                if lost_frames_counters[obj_key] > 3:
                    if obj_key == "ball":
                        ball_reintroduction.add_detection(detected_ball)
                        should_reintroduce, obj_data = ball_reintroduction.should_reintroduce()
                    elif obj_key == "rim":
                        rim_reintroduction.add_detection(detected_rim)
                        should_reintroduce, obj_data = rim_reintroduction.should_reintroduce()
                    else:  # Player
                        if obj_key not in player_reintroduction_trackers:
                            player_reintroduction_trackers[obj_key] = ReintroductionTracker(PLAYER_CONFIDENCE_THRESHOLD)
                        
                        # Find nearest detected player to last known position
                        if detected_players:
                            non_overlapping_players = [
                                p for p in detected_players 
                                if not is_detection_overlapping(p, trackers)
                            ]
                            
                            if non_overlapping_players:
                                best_player = non_overlapping_players[0]
                                player_reintroduction_trackers[obj_key].add_detection(best_player)
                            else:
                                player_reintroduction_trackers[obj_key].add_detection(None)
                        else:
                            player_reintroduction_trackers[obj_key].add_detection(None)
                        
                        should_reintroduce, obj_data = player_reintroduction_trackers[obj_key].should_reintroduce()
                    
                    if should_reintroduce:
                        print(f"Reintroducing {obj_key} at frame {frame_idx}")
                        trackers[obj_key] = ObjectTracker(obj_key, unique_id_counter)
                        unique_id_counter += 1
                        if frame_idx not in prompts_per_frame_index:
                            prompts_per_frame_index[frame_idx] = {}
                        prompts_per_frame_index[frame_idx][obj_key] = obj_data
                        lost_frames_counters[obj_key] = 0
                        if obj_key == "ball":
                            ball_reintroduction = ReintroductionTracker(BALL_CONFIDENCE_THRESHOLD)
                        elif obj_key == "rim":
                            rim_reintroduction = ReintroductionTracker(RIM_CONFIDENCE_THRESHOLD)
                            rim_reintroduction.required_consecutive_frames = RIM_REINTRODUCTION_FRAMES
                        else:
                            player_reintroduction_trackers[obj_key] = ReintroductionTracker(PLAYER_CONFIDENCE_THRESHOLD)
            
            # Encode frame data
            encoded_imgs_list, _, _ = sammodel.encode_image(
                frame,
                max_side_length=1024,
                use_square_sizing=True
            )
            
            # Handle initialization or reintroduction frame
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
            
            # First pass: generate all masks and update tracker information
            active_objects = []
            for obj_key, tracker in trackers.items():
                if not tracker.is_active:
                    continue
                    
                obj_memory = memory_per_obj_dict[obj_key]
                obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                    encoded_imgs_list, **obj_memory.to_dict()
                )
                
                # Update tracker status
                tracker.update_score(obj_score.item())
                if not tracker.is_active:
                    print(f"Removing {obj_key} due to consecutive low scores")
                    continue
                
                # Store memory
                obj_memory.store_result(frame_idx, mem_enc, obj_ptr)
                active_objects.append(obj_key)
                
                # Create mask and update tracker information
                obj_mask = torch.nn.functional.interpolate(
                    mask_preds[:, best_mask_idx, :, :],
                    size=combined_mask_result.shape,
                    mode="bilinear",
                    align_corners=False,
                )
                obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
                tracker.update_mask_and_centroid(obj_mask_binary)
                
                # Update visualization masks
                if obj_key == "ball":
                    ball_mask_result = np.bitwise_or(ball_mask_result, obj_mask_binary)
                elif obj_key == "rim":
                    rim_mask_result = np.bitwise_or(rim_mask_result, obj_mask_binary)
                else:
                    player_mask_result = np.bitwise_or(player_mask_result, obj_mask_binary)
                combined_mask_result = np.bitwise_or(combined_mask_result, obj_mask_binary)
            
            # Check for overlapping players and handle removal
            if frame_idx > 0:  # Skip first frame to allow initialization
                to_remove = check_for_overlaps(trackers)
                for obj_key in to_remove:
                    if trackers[obj_key].is_active:
                        print(f"Removing {obj_key} due to overlap at frame {frame_idx}")
                        trackers[obj_key].is_active = False
            
            # Log object tracking information
            log_object_tracking(trackers, frame_idx, tracking_data)
            
            # Create visualization frame - Modified to ensure ball is always on top
            vis_mask = np.zeros((*frame.shape[0:2], 3), dtype=np.uint8)
            
            # Layer the masks in specific order: rim -> players -> ball
            vis_mask[rim_mask_result] = [0, 255, 0]         # Green for rim (bottom layer)
            vis_mask[player_mask_result] = [255, 255, 255]  # White for players (middle layer)
            vis_mask[ball_mask_result] = [0, 165, 255]      # Orange for ball (top layer)
            
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
            active_players = len([k for k in active_objects if k not in ["ball", "rim"]])
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

        # Save JSON data
        os.makedirs(os.path.dirname(tracking_json_path), exist_ok=True)
        with open(tracking_json_path, 'w') as f:
            json.dump(tracking_data, f)
        print(f"Original video saved to: {output_original}")
        print(f"Mask video saved to: {output_masks}")
        print(f"Labels video saved to: {output_labels}")
        print(f"Tracking data saved to: {tracking_json_path}")

if __name__ == "__main__":
    main()