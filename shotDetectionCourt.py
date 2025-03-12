import numpy as np
import h5py
import cv2
from pathlib import Path
import os
import shutil
import json  # Added for JSON handling

def parse_csv_line(line):
    try:
        parts = line.strip().split(',')
        if len(parts) < 8:
            return None
        return {
            'frame': int(parts[0]),
            'timestamp': float(parts[1]),
            'confidence': float(parts[3]),
            'x1': float(parts[4]),
            'y1': float(parts[5]),
            'x2': float(parts[6]),
            'y2': float(parts[7])
        }
    except (ValueError, IndexError):
        return None

def load_shot_detections(csv_path):
    shots = []
    with open(csv_path, 'r') as f:
        for line in f:
            shot_data = parse_csv_line(line)
            if shot_data:
                shots.append(shot_data)
    shots.sort(key=lambda x: x['frame'])
    return shots

def find_sequence_starts(shots):
    sequence_starts = []
    i = 0
    
    while i < len(shots):
        current_frame = int(shots[i]['frame'])
        
        if shots[i]['confidence'] >= 0.45:
            sequence_starts.append(current_frame)
            while i < len(shots) and shots[i]['frame'] - current_frame <= 5:
                i += 1
            continue
            
        sequence = []
        j = i
        while j < len(shots) and shots[j]['frame'] - current_frame <= 4:
            sequence.append(int(shots[j]['frame']))
            j += 1
            
        if len(sequence) >= 3:
            consecutive = 1
            for k in range(len(sequence)-1):
                if sequence[k+1] == sequence[k] + 1:
                    consecutive += 1
                    if consecutive >= 3:
                        sequence_starts.append(current_frame)
                        break
                else:
                    consecutive = 1
                    
        if len(sequence) >= 4:
            sequence_starts.append(current_frame)
            
        while i < len(shots) and shots[i]['frame'] - current_frame <= 4:
            i += 1
            
        if i == j:
            i += 1
            
    return sorted(list(set(sequence_starts)))

def get_mask_from_player_h5(frame_group, object_key):
    try:
        obj = frame_group[object_key]
        rle_data = obj['mask_rle'][:]
        shape = tuple(obj['mask_shape'][:])
        
        mask = np.zeros(shape[0] * shape[1], dtype=bool)
        for start, length in rle_data:
            mask[start:start+length] = True
        return mask.reshape(shape)
    except KeyError:
        return None

def rle_decode(rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    for start, end in rle:
        mask[start:end] = True
    return mask.reshape(shape)

def find_shooter_and_jump(h5_file_path, shot_frame, max_lookback=200, distance_threshold=2, jump_detection_window=60):
    """
    Find the shooter and detect the jump frame with improved detection logic.
    
    Args:
        h5_file_path: Path to the player tracking h5 file
        shot_frame: Frame where the shot was detected
        max_lookback: Maximum number of frames to look back for shooter detection
        distance_threshold: Maximum distance between ball and player to be considered the shooter
        jump_detection_window: Maximum number of frames to look back for jump detection
        
    Returns:
        shooter_id: ID of the detected shooter
        jump_frame: Frame where the jump begins
        jump_position: Y position of the player's feet at jump
        jump_frames: List of all frames analyzed for jump detection
        object_id: The object_id attribute of the shooter
        detection_method: String indicating which method was used to detect the jump
    """
    shot_frame = int(shot_frame)
    closest_player_overall = None
    object_id_overall = None  # Store the object_id attribute
    player_positions = {}  # To track player positions over time
    ball_positions = {}    # To track ball positions
    ball_player_distances = {}  # Track the distance between player and ball
    detection_method = "unknown"  # New variable to track detection method
    player_object_ids = {}  # Store object_ids for players
    
    # First, track backward to find when the ball was last touched (original method)
    last_touch_frame = None
    with h5py.File(h5_file_path, 'r') as h5_file:
        for frame_idx in range(shot_frame, max(0, shot_frame - max_lookback), -1):
            frame_key = f"frame_{frame_idx:05d}"
            if frame_key not in h5_file:
                continue
                
            frame_group = h5_file[frame_key]
            
            if 'ball' not in frame_group:
                continue
                
            ball_mask = get_mask_from_player_h5(frame_group, 'ball')
            if ball_mask is None:
                continue
                
            ball_metadata = dict(frame_group['ball'].attrs)
            ball_x = int(float(np.float64(ball_metadata.get('centroid_x', 0))))
            ball_y = int(float(np.float64(ball_metadata.get('centroid_y', 0))))
            
            ball_pixels = np.where(ball_mask)
            if len(ball_pixels[0]) == 0:
                continue
                
            ball_points = np.column_stack((ball_pixels[0], ball_pixels[1]))
            
            min_distance = float('inf')
            closest_player = None
            closest_object_id = None
            
            for key in frame_group.keys():
                if not key.startswith('player'):
                    continue
                    
                player_mask = get_mask_from_player_h5(frame_group, key)
                if player_mask is None or not np.any(player_mask):
                    continue
                    
                # Get the object_id attribute
                player_obj = frame_group[key]
                player_attrs = dict(player_obj.attrs)
                object_id = player_attrs.get('object_id', None)
                
                player_pixels = np.where(player_mask)
                player_points = np.column_stack((player_pixels[0], player_pixels[1]))
                
                distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
                player_min_distance = np.min(distances)
                
                if player_min_distance < min_distance:
                    min_distance = player_min_distance
                    closest_player = key
                    closest_object_id = object_id
            
            if min_distance <= distance_threshold:
                # Found the last touch point
                last_touch_frame = frame_idx
                break
    
    # If we couldn't find a last touch point, return
    if last_touch_frame is None:
        return None, None, None, [], None, "no_last_touch_found"
    
    # Now starting from the last touch frame, look 5 frames backward and find the average closest player
    temporal_window = 5  # Look at 5 frames backward from last touch
    avg_distances = {}  # Track average distance for each player
    player_count = {}  # Count frames where each player was detected
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Start from last_touch_frame and go back 5 frames
        for frame_idx in range(last_touch_frame, max(0, last_touch_frame - temporal_window), -1):
            frame_key = f"frame_{frame_idx:05d}"
            if frame_key not in h5_file:
                continue
                
            frame_group = h5_file[frame_key]
            
            if 'ball' not in frame_group:
                continue
                
            ball_mask = get_mask_from_player_h5(frame_group, 'ball')
            if ball_mask is None:
                continue
                
            ball_pixels = np.where(ball_mask)
            if len(ball_pixels[0]) == 0:
                continue
                
            ball_points = np.column_stack((ball_pixels[0], ball_pixels[1]))
            
            # Calculate distance for each player to the ball
            for key in frame_group.keys():
                if not key.startswith('player'):
                    continue
                    
                player_mask = get_mask_from_player_h5(frame_group, key)
                if player_mask is None or not np.any(player_mask):
                    continue
                    
                # Get object_id
                player_obj = frame_group[key]
                player_attrs = dict(player_obj.attrs)
                object_id = player_attrs.get('object_id', None)
                player_object_ids[key] = object_id  # Store object_id for this player
                
                player_pixels = np.where(player_mask)
                player_points = np.column_stack((player_pixels[0], player_pixels[1]))
                
                distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
                player_min_distance = np.min(distances)
                
                # Update the average distances
                if key not in avg_distances:
                    avg_distances[key] = player_min_distance
                    player_count[key] = 1
                else:
                    avg_distances[key] = (avg_distances[key] * player_count[key] + player_min_distance) / (player_count[key] + 1)
                    player_count[key] += 1
    
    # Find the player with the lowest average distance
    min_avg_distance = float('inf')
    closest_player_overall = None
    
    for key, avg_distance in avg_distances.items():
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            closest_player_overall = key
    
    # Get object_id for the closest player
    if closest_player_overall is not None:
        object_id_overall = player_object_ids.get(closest_player_overall)
        detection_method = "temporal_average_distance"
    
    if closest_player_overall is None:
        return None, None, None, [], None, "no_shooter_found"
    
    # Now collect data for the shooter and ball across all relevant frames
    jump_frames = []
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        for frame_idx in range(shot_frame, max(0, shot_frame - jump_detection_window), -1):
            frame_key = f"frame_{frame_idx:05d}"
            if frame_key not in h5_file:
                continue
                
            frame_group = h5_file[frame_key]
            
            # We need both player and ball to be present
            if closest_player_overall not in frame_group or 'ball' not in frame_group:
                continue
                
            # Get player mask and position
            player_mask = get_mask_from_player_h5(frame_group, closest_player_overall)
            if player_mask is None or not np.any(player_mask):
                continue
                
            # Get ball mask and position
            ball_mask = get_mask_from_player_h5(frame_group, 'ball')
            if ball_mask is None or not np.any(ball_mask):
                continue
            
            # Get player's foot position (average of bottom 20% of the mask)
            player_y_positions = np.where(player_mask)[0]
            if len(player_y_positions) > 0:
                # Sort y-positions in descending order (bottom to top)
                sorted_y = np.sort(player_y_positions)[::-1]
                # Take bottom 20% of pixels
                bottom_pixels = sorted_y[:max(1, int(len(sorted_y) * 0.1))]
                # Calculate average position
                foot_position = np.mean(bottom_pixels)
                player_positions[frame_idx] = foot_position
            
            # Get ball position
            ball_pixels = np.where(ball_mask)
            if len(ball_pixels[0]) > 0:
                ball_y = np.mean(ball_pixels[0])
                ball_positions[frame_idx] = ball_y
                
                # Calculate minimum distance between player and ball
                player_pixels = np.where(player_mask)
                player_points = np.column_stack((player_pixels[0], player_pixels[1]))
                ball_points = np.column_stack((ball_pixels[0], ball_pixels[1]))
                
                distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
                min_distance = np.min(distances)
                
                ball_player_distances[frame_idx] = min_distance
                
            jump_frames.append(frame_idx)
    
    if not player_positions or not ball_positions:
        return closest_player_overall, None, None, [], object_id_overall, "no_position_data"
    
    # Sort frames to ensure we're analyzing in chronological order
    sorted_frames = sorted(jump_frames)
    
    # Find frames where player has the ball (very small distance between them)
    POSSESSION_DISTANCE_THRESHOLD = 15  # Maximum distance to consider player has possession of ball
    possession_frames = [frame for frame in sorted_frames 
                        if frame in ball_player_distances 
                        and ball_player_distances[frame] <= POSSESSION_DISTANCE_THRESHOLD]
    
    if not possession_frames:
        return closest_player_overall, None, None, jump_frames, object_id_overall, "no_ball_possession"
    
    # Only analyze frames where player has possession
    valid_frames = sorted(possession_frames)
    
    # Check if we have enough frames for analysis
    if len(valid_frames) < 3:
        return closest_player_overall, None, None, jump_frames, object_id_overall, "insufficient_possession_frames"
    
    # Calculate velocities for player feet (y-position)
    # Negative velocity means moving up in image coordinates
    player_velocities = []
    for i in range(1, len(valid_frames)):
        prev_frame = valid_frames[i-1]
        curr_frame = valid_frames[i]
        
        if prev_frame in player_positions and curr_frame in player_positions:
            prev_pos = player_positions[prev_frame]
            curr_pos = player_positions[curr_frame]
            
            frame_diff = curr_frame - prev_frame
            velocity = (curr_pos - prev_pos) / frame_diff
            
            player_velocities.append((curr_frame, velocity))
    
    if not player_velocities:
        return closest_player_overall, None, None, jump_frames, object_id_overall, "could_not_calculate_velocities"
    
    # Find the start of the first upward movement sequence
    # (Negative velocity means moving up in image coordinates)
    upward_movement_sequences = []
    current_sequence = []
    
    # Threshold for significant upward movement
    UPWARD_VELOCITY_THRESHOLD = -0.5  # Negative because moving up means decreasing y-value
    
    for frame, velocity in player_velocities:
        if velocity < UPWARD_VELOCITY_THRESHOLD:
            # Player is moving upward
            if not current_sequence:
                # This is the start of a new sequence
                current_sequence.append((frame, velocity))
            else:
                # Continue the current sequence
                current_sequence.append((frame, velocity))
        else:
            # Player is not moving upward
            if current_sequence:
                # End of a sequence
                if len(current_sequence) >= 2:  # Require at least 2 consecutive frames of upward movement
                    upward_movement_sequences.append(current_sequence)
                current_sequence = []
    
    # Don't forget to add the last sequence if it's still open
    if current_sequence and len(current_sequence) >= 2:
        upward_movement_sequences.append(current_sequence)
    
    # Find the first sequence of upward movement
    if upward_movement_sequences:
        # Get the first frame of the first upward movement sequence
        first_sequence = upward_movement_sequences[0]
        jump_frame = first_sequence[0][0]
        jump_velocity = first_sequence[0][1]
        
        print(f"Jump detected at the FIRST frame of upward movement sequence:")
        print(f"  - Frame: {jump_frame}")
        print(f"  - Velocity: {jump_velocity:.2f} pixels/frame")
        print(f"  - Sequence length: {len(first_sequence)} frames")
        
        detection_method = "first_upward_movement_frame"
    else:
        # Fallback: Find the frame with the most negative velocity (strongest upward movement)
        player_velocities.sort(key=lambda x: x[1])  # Sort by velocity (most negative first)
        
        if player_velocities[0][1] < UPWARD_VELOCITY_THRESHOLD:
            jump_frame = player_velocities[0][0]
            jump_velocity = player_velocities[0][1]
            
            print(f"Jump detected using most significant upward velocity:")
            print(f"  - Frame: {jump_frame}")
            print(f"  - Velocity: {jump_velocity:.2f} pixels/frame")
            
            detection_method = "most_significant_upward_velocity"
        else:
            # Last resort: use the first possession frame
            jump_frame = valid_frames[0]
            
            print(f"No clear upward movement detected. Using first possession frame:")
            print(f"  - Frame: {jump_frame}")
            
            detection_method = "first_possession_frame"
    
    # Get the final foot position at the jump frame
    jump_position = player_positions.get(jump_frame) if jump_frame else None
    
    return closest_player_overall, jump_frame, jump_position, jump_frames, object_id_overall, detection_method

def check_three_point_shot(court_h5_path, player_h5_path, frame_num, shooter_id, proximity_threshold=30):
    """
    Determine if a shot is taken from behind the 3-point line using a geometric approach.
    
    Args:
        court_h5_path: Path to the court tracking h5 file
        player_h5_path: Path to the player tracking h5 file
        frame_num: Frame number to analyze
        shooter_id: ID of the shooter player
        proximity_threshold: Maximum distance in pixels to consider player near 3-point line
        
    Returns:
        is_three_point: Boolean indicating if shot is from behind 3-point line
        proximity_to_line: Distance to 3-point line in pixels
        three_point_analysis: Dictionary with details of the analysis
    """
    # Default return values
    is_three_point = False
    proximity_to_line = float('inf')
    three_point_analysis = {
        "is_near_line": False,
        "is_three_point": False,
        "court_orientation": "unknown",
        "distance_to_line": float('inf')
    }
    
    # Get court masks
    court_masks = {}
    court_dimensions = None
    
    try:
        with h5py.File(court_h5_path, 'r') as court_h5:
            # Find the closest court frame
            court_frame_key = find_closest_court_frame(court_h5_path, frame_num)
            
            frames_group = court_h5
            if 'frames' in court_h5:
                frames_group = court_h5['frames']
            
            if court_frame_key is not None and court_frame_key in frames_group:
                frame = frames_group[court_frame_key]
                
                # Store court dimensions for scaling
                for detection_key in frame.keys():
                    detection = frame[detection_key]
                    rle_data = detection['rle'][:]
                    mask_shape = tuple(detection['rle'].attrs['shape'])
                    court_dimensions = mask_shape
                    break
                
                # Get all court component masks
                for detection_key in frame.keys():
                    detection = frame[detection_key]
                    
                    class_id = detection.attrs.get('class_id', detection.attrs.get('class', 0))
                    
                    rle_data = detection['rle'][:]
                    mask_shape = tuple(detection['rle'].attrs['shape'])
                    mask = rle_decode(rle_data, mask_shape)
                    
                    if class_id not in court_masks:
                        court_masks[class_id] = mask
                    else:
                        # Combine masks of the same class
                        court_masks[class_id] = np.logical_or(court_masks[class_id], mask)
    except Exception as e:
        print(f"Error loading court data: {e}")
        return is_three_point, proximity_to_line, three_point_analysis
    
    # Get player foot position and mask
    player_foot_position = None
    player_foot_mask = None
    player_dimensions = None
    
    try:
        with h5py.File(player_h5_path, 'r') as player_h5:
            player_frame_key = f"frame_{frame_num:05d}"
            if player_frame_key in player_h5:
                frame_group = player_h5[player_frame_key]
                
                # Store player dimensions for scaling
                for key in frame_group.keys():
                    if key.startswith('player'):
                        mask = get_mask_from_player_h5(frame_group, key)
                        if mask is not None:
                            player_dimensions = mask.shape
                            break
                
                # Get shooter mask if available
                if shooter_id in frame_group:
                    player_mask = get_mask_from_player_h5(frame_group, shooter_id)
                    if player_mask is not None and np.any(player_mask):
                        player_dimensions = player_mask.shape
                        
                        # Get bottom 20% of player mask (feet)
                        player_y_positions = np.where(player_mask)[0]
                        player_x_positions = np.where(player_mask)[1]
                        
                        if len(player_y_positions) > 0:
                            # Sort y-positions in descending order (bottom to top)
                            sorted_indices = np.argsort(player_y_positions)[::-1]
                            
                            # Take bottom 10% of pixels (feet)
                            bottom_indices = sorted_indices[:max(1, int(len(sorted_indices) * 0.1))]
                            bottom_y = player_y_positions[bottom_indices]
                            bottom_x = player_x_positions[bottom_indices]
                            
                            # Create a mask for just the bottom 10%
                            player_foot_mask = np.zeros_like(player_mask)
                            player_foot_mask[bottom_y, bottom_x] = True
                            
                            # Calculate average position for reference
                            avg_y = int(np.mean(bottom_y))
                            avg_x = int(np.mean(bottom_x))
                            player_foot_position = (avg_y, avg_x)
    except Exception as e:
        print(f"Error loading player data: {e}")
        return is_three_point, proximity_to_line, three_point_analysis
    
    # If we couldn't get necessary data, return early
    if len(court_masks) == 0 or player_foot_mask is None or player_foot_position is None:
        return is_three_point, proximity_to_line, three_point_analysis
    
    # Scale court masks to match player dimensions if needed
    if court_dimensions and player_dimensions:
        if court_dimensions[0] != player_dimensions[0] or court_dimensions[1] != player_dimensions[1]:
            scaled_court_masks = {}
            for class_id, mask in court_masks.items():
                scaled_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (player_dimensions[1], player_dimensions[0])
                ).astype(bool)
                scaled_court_masks[class_id] = scaled_mask
            court_masks = scaled_court_masks
    
    # Get three-point line and keyway masks
    three_point_mask = court_masks.get(2, None)  # Three-point line (usually class_id 2)
    keyway_mask = court_masks.get(1, None)       # Keyway (usually class_id 1)
    
    if three_point_mask is None or keyway_mask is None:
        return is_three_point, proximity_to_line, three_point_analysis
    
    # ----- Determine court orientation based on keyway and three-point line positions -----
    # Get centers of keyway and three-point line
    keyway_y, keyway_x = np.where(keyway_mask)
    three_pt_y, three_pt_x = np.where(three_point_mask)
    
    if len(keyway_y) == 0 or len(three_pt_y) == 0:
        return is_three_point, proximity_to_line, three_point_analysis
    
    keyway_center_x = np.mean(keyway_x)
    keyway_center_y = np.mean(keyway_y)
    three_pt_center_x = np.mean(three_pt_x)
    three_pt_center_y = np.mean(three_pt_y)
    
    # Determine court orientation
    court_orientation = "unknown"
    if keyway_center_x < three_pt_center_x:
        court_orientation = "looking_right"  # Basket is on the right side
    else:
        court_orientation = "looking_left"   # Basket is on the left side
    
    three_point_analysis["court_orientation"] = court_orientation
    
    # ----- Find closest point on three-point line to player's feet -----
    player_y, player_x = player_foot_position
    min_distance = float('inf')
    closest_point = None
    
    # To avoid excessive computation, sample points from the three-point line
    three_pt_points = np.column_stack((three_pt_y, three_pt_x))
    if len(three_pt_points) > 1000:
        # Randomly sample 1000 points
        indices = np.random.choice(len(three_pt_points), 1000, replace=False)
        three_pt_points = three_pt_points[indices]
    
    for point_y, point_x in three_pt_points:
        distance = np.sqrt((point_y - player_y)**2 + (point_x - player_x)**2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (point_y, point_x)
    
    if closest_point is None:
        return is_three_point, proximity_to_line, three_point_analysis
    
    proximity_to_line = min_distance
    three_point_analysis["distance_to_line"] = min_distance
    
    # ----- Determine if player is on the outside of the three-point line -----
    # For this, we need to check if the player is on the opposite side of the line from the keyway
    
    if min_distance <= proximity_threshold:
        three_point_analysis["is_near_line"] = True
        
        # Create a vector from the closest point on the three-point line to the keyway center
        # This vector points toward the inside of the court
        line_to_keyway_y = keyway_center_y - closest_point[0]
        line_to_keyway_x = keyway_center_x - closest_point[1]
        
        # Create a vector from the closest point to the player's feet
        line_to_player_y = player_y - closest_point[0]
        line_to_player_x = player_x - closest_point[1]
        
        # Check if these vectors point in opposite directions using dot product
        # If dot product is negative, the vectors point in opposite directions
        # This means the player is on the outside of the three-point line
        dot_product = (line_to_keyway_y * line_to_player_y) + (line_to_keyway_x * line_to_player_x)
        
        # Normalize by magnitudes to get cosine of angle between vectors
        magnitude_keyway = np.sqrt(line_to_keyway_y**2 + line_to_keyway_x**2)
        magnitude_player = np.sqrt(line_to_player_y**2 + line_to_player_x**2)
        
        if magnitude_keyway > 0 and magnitude_player > 0:
            normalized_dot = dot_product / (magnitude_keyway * magnitude_player)
            
            # If dot product is negative, the player is outside the three-point line
            if normalized_dot < 0:
                is_three_point = True
                three_point_analysis["is_three_point"] = True
                three_point_analysis["normalized_dot_product"] = normalized_dot
    
    return is_three_point, proximity_to_line, three_point_analysis

def create_visualization_with_court(court_h5_path, player_h5_path, frame_num, shooter_id, output_path, 
                                   is_jump_frame=False, jersey_data=None, detection_method=None,
                                   is_three_point=None, three_point_analysis=None):
    default_height, default_width = 1080, 1920
    combined_canvas = np.zeros((default_height, default_width, 3), dtype=np.uint8)
    
    try:
        with h5py.File(court_h5_path, 'r') as court_h5:
            court_colors = {
                0: (200, 0, 0),   # Blue - Boundary
                1: (150, 0, 150),  # Purple - Key
                2: (0, 0, 200),    # Red - Three point
                3: (255, 255, 0)   # Yellow - Other
            }
            
            court_frame_key = find_closest_court_frame(court_h5_path, frame_num)
            
            frames_group = court_h5
            if 'frames' in court_h5:
                frames_group = court_h5['frames']
            
            if court_frame_key is not None and court_frame_key in frames_group:
                frame = frames_group[court_frame_key]
                
                for detection_key in frame.keys():
                    detection = frame[detection_key]
                    
                    class_id = detection.attrs.get('class_id', detection.attrs.get('class', 0))
                    
                    rle_data = detection['rle'][:]
                    mask_shape = tuple(detection['rle'].attrs['shape'])
                    mask = rle_decode(rle_data, mask_shape)
                    
                    color = court_colors.get(class_id, (128, 128, 128))
                    
                    if mask_shape[0] != default_height or mask_shape[1] != default_width:
                        temp_canvas = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
                        temp_canvas[mask] = color
                        
                        temp_canvas = cv2.resize(temp_canvas, (default_width, default_height))
                        
                        resized_mask = np.any(temp_canvas > 0, axis=2)
                        for c in range(3):
                            combined_canvas[:, :, c][resized_mask] = color[c]
                    else:
                        combined_canvas[mask] = color
    
    except Exception as e:
        print(f"Error processing court data: {e}")
    
    # Store the shooter's object_id to add jersey number if available
    shooter_object_id = None
    
    # Store bottom 20% mask for visualization enhancement
    bottom_mask_20_percent = None
    bottom_area_bbox = None
    
    try:
        with h5py.File(player_h5_path, 'r') as player_h5:
            player_frame_key = f"frame_{frame_num:05d}"
            if player_frame_key in player_h5:
                frame_group = player_h5[player_frame_key]
                
                for obj_key in frame_group.keys():
                    mask = get_mask_from_player_h5(frame_group, obj_key)
                    if mask is None:
                        continue
                    
                    # Get object_id if this is the shooter
                    if obj_key == shooter_id:
                        player_obj = frame_group[obj_key]
                        player_attrs = dict(player_obj.attrs)
                        shooter_object_id = player_attrs.get('object_id', None)
                    
                    # Determine coloring:
                    # - All players are white if shooter_id is None (pre-jump frames)
                    # - Yellow for shooter at jump frame
                    # - Green for shooter in other frames after jump
                    # - White for non-shooters
                    if shooter_id is None:
                        color = (255, 255, 255)  # Everyone is white in pre-jump frames
                    else:
                        color = (0, 255, 255) if (obj_key == shooter_id and is_jump_frame) else \
                               (0, 255, 0) if obj_key == shooter_id else \
                               (255, 255, 255)
                    
                    if mask.shape[0] != combined_canvas.shape[0] or mask.shape[1] != combined_canvas.shape[1]:
                        temp_canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        temp_canvas[mask] = color
                        
                        temp_canvas = cv2.resize(temp_canvas, (combined_canvas.shape[1], combined_canvas.shape[0]))
                        
                        resized_mask = np.any(temp_canvas > 0, axis=2)
                        for c in range(3):
                            combined_canvas[:, :, c][resized_mask] = color[c]
                    else:
                        for c in range(3):
                            combined_canvas[:, :, c][mask] = color[c]
                    
                    # Add player ID
                    obj_metadata = dict(frame_group[obj_key].attrs)
                    text_x = int(float(obj_metadata['centroid_x']))
                    text_y = int(float(obj_metadata['centroid_y']))
                    
                    cv2.putText(combined_canvas, obj_key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 0, 0), 2)
                    cv2.putText(combined_canvas, obj_key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 255, 255), 1)
                    
                    # Create enhanced visualization for the bottom 20% area of the shooter
                    if obj_key == shooter_id:
                        player_pixels = np.where(mask)
                        if len(player_pixels[0]) > 0:
                            # Sort y-positions in descending order (bottom to top)
                            y_positions = player_pixels[0]
                            x_positions = player_pixels[1]
                            sorted_indices = np.argsort(y_positions)[::-1]
                            
                            # Take bottom 20% of pixels
                            bottom_indices = sorted_indices[:max(1, int(len(sorted_indices) * 0.1))]
                            bottom_y = y_positions[bottom_indices]
                            bottom_x = x_positions[bottom_indices]
                            
                            # Create a mask for just the bottom 20%
                            bottom_mask_20_percent = np.zeros_like(mask)
                            bottom_mask_20_percent[bottom_y, bottom_x] = True
                            
                            # Calculate average position
                            avg_y = int(np.mean(bottom_y))
                            avg_x = int(np.mean(bottom_x))
                            
                            # Get bounding box of the area for visualization
                            min_x = np.min(bottom_x)
                            max_x = np.max(bottom_x)
                            min_y = np.min(bottom_y)
                            max_y = np.max(bottom_y)
                            bottom_area_bbox = (min_x, min_y, max_x, max_y)
                            
                            # For jump frames, apply enhanced visualization
                            if is_jump_frame:
                                # Apply a more vibrant cyan color to the bottom 20% area
                                if bottom_mask_20_percent.shape[0] != combined_canvas.shape[0] or bottom_mask_20_percent.shape[1] != combined_canvas.shape[1]:
                                    temp_canvas = np.zeros((bottom_mask_20_percent.shape[0], bottom_mask_20_percent.shape[1], 3), dtype=np.uint8)
                                    temp_canvas[bottom_mask_20_percent] = (0, 255, 255)  # Cyan
                                    
                                    temp_canvas = cv2.resize(temp_canvas, (combined_canvas.shape[1], combined_canvas.shape[0]))
                                    
                                    resized_mask = np.any(temp_canvas > 0, axis=2)
                                    for c in range(3):
                                        # Apply a semi-transparent overlay (75% opacity)
                                        combined_canvas[:, :, c][resized_mask] = combined_canvas[:, :, c][resized_mask] * 0.25 + temp_canvas[:, :, c][resized_mask] * 0.75
                                else:
                                    for c in range(3):
                                        # Apply a semi-transparent overlay (75% opacity)
                                        combined_canvas[:, :, c][bottom_mask_20_percent] = combined_canvas[:, :, c][bottom_mask_20_percent] * 0.25 + (0 if c == 0 else 255) * 0.75
                                
                                # Draw a thicker red rectangle around the bottom 20% area
                                if bottom_area_bbox:
                                    min_x, min_y, max_x, max_y = bottom_area_bbox
                                    cv2.rectangle(combined_canvas, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
                                
                                # Add text indicating this is the jump point with larger font
                                if avg_x and avg_y:
                                    cv2.putText(combined_canvas, "JUMP AREA", (avg_x, max_y + 20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    except Exception as e:
        print(f"Error processing player data: {e}")
    
    # Add frame number and phase indication
    frame_text = f"Frame: {frame_num}"
    if is_jump_frame:
        frame_text += " (JUMP FRAME)"
        if detection_method:
            frame_text += f" - Method: {detection_method}"
    elif shooter_id is None:
        frame_text += " (PRE-JUMP)"
    else:
        frame_text += " (SHOT SEQUENCE)"
    
    # Add jersey number if available
    if shooter_object_id is not None and jersey_data and str(shooter_object_id) in jersey_data:
        jersey_number = jersey_data[str(shooter_object_id)]['number']
        frame_text += f" - Player #{jersey_number}"
    
    # Add three-point shot information if available
    if is_three_point is not None:
        if is_three_point:
            frame_text += " - 3PT SHOT"
        else:
            frame_text += " - 2PT SHOT"
    
    cv2.putText(combined_canvas, frame_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), combined_canvas)
    return combined_canvas

def find_closest_court_frame(court_h5_path, target_frame):
    target_frame = int(target_frame)
    
    try:
        with h5py.File(court_h5_path, 'r') as court_h5:
            frames_group = None
            if 'frames' in court_h5:
                frames_group = court_h5['frames']
            else:
                direct_frame_keys = [k for k in court_h5.keys() if k.startswith('frame')]
                if direct_frame_keys:
                    frames_group = court_h5
            
            if frames_group is None:
                return None
                
            frame_keys = list(frames_group.keys())
            if not frame_keys:
                return None
            
            exact_match_keys = [
                f"frame_{target_frame:05d}",
                f"frame_{target_frame}",
                f"frame{target_frame}"
            ]
            
            for key in exact_match_keys:
                if key in frames_group:
                    return key
            
            frame_numbers = []
            for key in frame_keys:
                if key.startswith('frame_'):
                    try:
                        num = int(key.split('frame_')[1])
                        frame_numbers.append((key, num))
                    except (ValueError, IndexError):
                        pass
                elif key.startswith('frame'):
                    try:
                        num = int(key.replace('frame', ''))
                        frame_numbers.append((key, num))
                    except ValueError:
                        pass
            
            if not frame_numbers:
                return frame_keys[0]
                
            frame_numbers.sort(key=lambda x: abs(x[1] - target_frame))
            return frame_numbers[0][0]
            
    except Exception as e:
        print(f"Error finding closest court frame: {e}")
        return None

def create_jump_sequence_video(court_h5_path, player_h5_path, jump_frame, shot_frame, shooter_id, output_path, fps=10, pre_jump_frames=20, jersey_data=None, detection_method=None):
    # Include frames before the jump for context
    start_frame = max(1, jump_frame - pre_jump_frames)
    
    # Get all frames from pre-jump through shot
    frames = list(range(start_frame, shot_frame + 1))
    
    # Create temporary directory for frames
    temp_dir = Path('temp_frames')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # First check if the shot is a three-pointer
        three_point_check = check_three_point_shot(court_h5_path, player_h5_path, jump_frame, shooter_id)
        is_three_point, proximity_to_line, three_point_analysis = three_point_check
        
        print(f"Three-point shot analysis at jump frame {jump_frame}:")
        print(f"  Is three-point shot: {is_three_point}")
        print(f"  Distance to 3PT line: {proximity_to_line:.2f} pixels")
        print(f"  Analysis details: {three_point_analysis}")
        
        # Generate frames
        frame_paths = []
        for i, frame_num in enumerate(frames):
            is_jump_frame = (frame_num == jump_frame)
            is_pre_jump = (frame_num < jump_frame)
            frame_path = temp_dir / f'frame_{i:05d}.png'
            frame_paths.append(frame_path)
            
            # Only highlight the shooter after the jump begins
            current_shooter_id = None if is_pre_jump else shooter_id
            
            # Include detection method for visualization
            # Also include three-point shot info if this is the jump frame
            current_is_three_point = is_three_point if is_jump_frame else None
            current_three_point_analysis = three_point_analysis if is_jump_frame else None
            
            create_visualization_with_court(
                court_h5_path, 
                player_h5_path, 
                frame_num, 
                current_shooter_id, 
                frame_path, 
                is_jump_frame, 
                jersey_data,
                detection_method if is_jump_frame else None,
                current_is_three_point,
                current_three_point_analysis
            )
        
        # Check if we have any frames
        if not frame_paths:
            print("No frames were generated for the video.")
            return False
        
        # Get frame dimensions from the first frame
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            print(f"Could not read frame: {frame_paths[0]}")
            return False
            
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        
        return True
        
    except Exception as e:
        print(f"Error creating sequence video: {e}")
        return False
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            for file in temp_dir.glob('*.png'):
                file.unlink()
            temp_dir.rmdir()

def find_sequence_starts(shots, min_frame_gap=100):
    """
    Find the starting frames of shot sequences while ensuring a minimum gap between sequences.
    Prioritizes sequences with higher confidence when multiple occur within the minimum gap.
    
    Args:
        shots: List of shot detection dictionaries
        min_frame_gap: Minimum number of frames that must separate two shot sequences
        
    Returns:
        List of frame numbers where shot sequences start
    """
    if not shots:
        return []
        
    # Group shots by proximity into potential sequences
    potential_sequences = []
    current_sequence = [shots[0]]
    
    for i in range(1, len(shots)):
        current_shot = shots[i]
        prev_shot = shots[i-1]
        
        # If this shot is close to the previous one, add it to the current sequence
        if current_shot['frame'] - prev_shot['frame'] <= 5:
            current_sequence.append(current_shot)
        else:
            # This shot is too far from the previous one, so it starts a new sequence
            if current_sequence:
                potential_sequences.append(current_sequence)
            current_sequence = [current_shot]
    
    # Don't forget to add the last sequence
    if current_sequence:
        potential_sequences.append(current_sequence)
    
    # Analyze each potential sequence to determine if it's a valid sequence
    valid_sequences = []
    
    for sequence in potential_sequences:
        # Apply the existing sequence validation logic
        if check_sequence_validity(sequence):
            # Calculate the average confidence of this sequence
            avg_confidence = sum(shot['confidence'] for shot in sequence) / len(sequence)
            start_frame = sequence[0]['frame']
            valid_sequences.append((start_frame, avg_confidence))
    
    # Sort by frame number
    valid_sequences.sort(key=lambda x: x[0])
    
    # Apply minimum frame gap, prioritizing sequences with higher confidence
    filtered_sequences = []
    i = 0
    while i < len(valid_sequences):
        current_frame, current_confidence = valid_sequences[i]
        
        # Find all sequences within min_frame_gap of the current one
        j = i + 1
        conflicting_sequences = []
        
        while j < len(valid_sequences) and valid_sequences[j][0] - current_frame < min_frame_gap:
            conflicting_sequences.append(valid_sequences[j])
            j += 1
        
        if not conflicting_sequences:
            # No conflicts, add this sequence
            filtered_sequences.append(current_frame)
            i += 1
        else:
            # Find the sequence with highest confidence among the current and all conflicting ones
            all_candidates = [(current_frame, current_confidence)] + conflicting_sequences
            best_sequence = max(all_candidates, key=lambda x: x[1])
            
            # Add only the best sequence
            filtered_sequences.append(best_sequence[0])
            
            # Skip over all the conflicting sequences
            i = j
    
    return filtered_sequences

def check_sequence_validity(sequence):
    """
    Check if a sequence of shots is valid based on existing criteria:
    1. If any shot has confidence >= 0.45
    2. If at least 3 consecutive frames have detections
    3. If the sequence has at least 4 frames with detections
    
    Args:
        sequence: List of shot detection dictionaries
        
    Returns:
        Boolean indicating if the sequence is valid
    """
    # Check if any shot has high confidence
    if any(shot['confidence'] >= 0.45 for shot in sequence):
        return True
    
    # Check for 3 consecutive frames
    if len(sequence) >= 3:
        for i in range(len(sequence) - 2):
            frame1 = sequence[i]['frame']
            frame2 = sequence[i+1]['frame']
            frame3 = sequence[i+2]['frame']
            
            if frame2 == frame1 + 1 and frame3 == frame2 + 1:
                return True
    
    # Check if sequence has at least 4 frames
    if len(sequence) >= 4:
        return True
    
    return False

def main():
    fileName = "4thQMackay1"
    court_h5_path = f'courtTrackingOutputs/{fileName}.h5'
    player_h5_path = f'trackingOutputs/{fileName}.h5'
    shot_csv_path = f'madeShotCSV/{fileName}.csv'
    jersey_json_path = f'jerseyNumbers/{fileName}_jersey.json'  # Updated path with _jersey
    
    # Create output directories
    output_dir = Path('madeShots')
    output_dir.mkdir(exist_ok=True)
    
    jump_detection_dir = Path('jumpDetection')
    jump_detection_dir.mkdir(exist_ok=True)
    
    sequences_dir = Path('shotSequences')
    sequences_dir.mkdir(exist_ok=True)
    
    # Create a new directory for detection method logs
    detection_logs_dir = Path('detectionLogs')
    detection_logs_dir.mkdir(exist_ok=True)
    
    # Create a directory for three-point analysis
    three_point_dir = Path('threePointAnalysis')
    three_point_dir.mkdir(exist_ok=True)
    
    # Create a log file for detection methods with 3-point information
    log_file = detection_logs_dir / f'{fileName}_detection_methods.csv'
    with open(log_file, 'w') as f:
        f.write("shot_frame,jump_frame,shooter_id,detection_method,jersey_number,is_three_point,distance_to_line\n")
    
    # Load jersey number data
    jersey_data = None
    try:
        with open(jersey_json_path, 'r') as f:
            jersey_data = json.load(f)
        print(f"Successfully loaded jersey numbers from {jersey_json_path}")
    except Exception as e:
        print(f"Error loading jersey numbers: {e}")
    
    # Load shot detections
    shots = load_shot_detections(shot_csv_path)
    
    # Use the improved function with 100-frame minimum gap
    sequence_starts = find_sequence_starts(shots, min_frame_gap=100)
    
    print(f"Found {len(sequence_starts)} shot sequences with minimum 100-frame gap")
    
    for shot_frame in sequence_starts:
        print(f"\n\nProcessing shot sequence starting at frame {shot_frame}")
        
        # Find shooter and detect jump - now with detection_method
        shooter_id, jump_frame, jump_position, all_frames, object_id, detection_method = find_shooter_and_jump(player_h5_path, shot_frame)
        
        # Get jersey number if available
        jersey_number = "unknown"
        if shooter_id is not None:
            print(f"Found shooter {shooter_id}")
            
            # Check if this object_id exists in the jersey data
            if jersey_data and object_id is not None:
                object_id_str = str(object_id)
                if object_id_str in jersey_data:
                    jersey_number = jersey_data[object_id_str]['number']
                    confidence = jersey_data[object_id_str]['total_confidence']
                    print(f"SHOOTER JERSEY NUMBER: #{jersey_number} (Confidence: {confidence:.2f})")
                else:
                    print(f"No jersey number found for object ID {object_id}")
            else:
                print("No jersey data available or object_id not found")
        
        # Check if the shot is a three-pointer (if we have a valid jump frame)
        is_three_point = False
        proximity_to_line = float('inf')
        three_point_analysis = None
        
        # Only check for three-point if we have both a shooter and a jump frame
        if shooter_id is not None and jump_frame is not None:
            try:
                # Only analyze the jump frame for three-point shot detection
                three_point_check = check_three_point_shot(court_h5_path, player_h5_path, jump_frame, shooter_id)
                is_three_point, proximity_to_line, three_point_analysis = three_point_check
                
                print(f"Three-point shot analysis:")
                print(f"  Is three-point shot: {is_three_point}")
                print(f"  Distance to 3PT line: {proximity_to_line:.2f} pixels")
                print(f"  Analysis details: {three_point_analysis}")
                
                # Save a separate visualization for the three-point analysis
                three_point_viz_path = three_point_dir / f'three_point_analysis_{jump_frame:05d}.png'
                # Create enhanced visualization showing 3-point line and player position
                create_visualization_with_court(
                    court_h5_path, 
                    player_h5_path, 
                    jump_frame, 
                    shooter_id, 
                    three_point_viz_path, 
                    True, 
                    jersey_data, 
                    detection_method,
                    is_three_point,
                    three_point_analysis
                )
            except Exception as e:
                # If there's an error in three-point analysis, log it but continue processing
                print(f"Error analyzing three-point shot: {e}")
                is_three_point = False
                proximity_to_line = float('inf')
                three_point_analysis = None
        
        # Log the detection information with 3-point data
        with open(log_file, 'a') as f:
            three_point_str = "Yes" if is_three_point else "No"
            distance_str = f"{proximity_to_line:.2f}" if proximity_to_line != float('inf') else "N/A"
            f.write(f"{shot_frame},{jump_frame if jump_frame else 'None'},{shooter_id if shooter_id else 'None'},{detection_method},{jersey_number},{three_point_str},{distance_str}\n")
        
        # Create a detailed log for this shot
        shot_log_file = detection_logs_dir / f'shot_{shot_frame}_details.txt'
        with open(shot_log_file, 'w') as f:
            f.write(f"Shot Frame: {shot_frame}\n")
            f.write(f"Shooter ID: {shooter_id if shooter_id else 'Not detected'}\n")
            f.write(f"Jump Frame: {jump_frame if jump_frame else 'Not detected'}\n")
            f.write(f"Detection Method: {detection_method}\n")
            f.write(f"Jersey Number: {jersey_number}\n")
            f.write(f"Three-Point Shot: {three_point_str}\n")
            f.write(f"Distance to 3PT Line: {distance_str} pixels\n\n")
            
            if jump_frame and jump_position:
                f.write(f"Jump Position (y-coordinate): {jump_position}\n")
            
            if three_point_analysis:
                f.write("\nThree-Point Analysis Details:\n")
                for key, value in three_point_analysis.items():
                    f.write(f"  - {key}: {value}\n")
            
            f.write(f"\nAll analyzed frames: {all_frames}\n")
        
        if shooter_id is not None:
            if jump_frame is not None:
                print(f"Detected jump at frame {jump_frame}, foot position: {jump_position}")
                print(f"Detection method: {detection_method}")
                
                # Add detection method and 3-point info to the visualization filename
                three_point_suffix = "3PT" if is_three_point else "2PT"
                jump_output_path = jump_detection_dir / f'jump_frame_{jump_frame:05d}_{detection_method}_{three_point_suffix}.png'
                create_visualization_with_court(
                    court_h5_path, 
                    player_h5_path, 
                    jump_frame, 
                    shooter_id, 
                    jump_output_path, 
                    True, 
                    jersey_data, 
                    detection_method,
                    is_three_point,
                    three_point_analysis
                )
                
                # Create a video sequence from 20 frames before jump to shot
                sequence_output_path = sequences_dir / f'sequence_{jump_frame}_to_{shot_frame}_{detection_method}_{three_point_suffix}.mp4'
                success = create_jump_sequence_video(
                    court_h5_path, 
                    player_h5_path, 
                    jump_frame, 
                    shot_frame, 
                    shooter_id, 
                    sequence_output_path, 
                    fps=10, 
                    pre_jump_frames=20, 
                    jersey_data=jersey_data, 
                    detection_method=detection_method
                )
                
                if success:
                    print(f"Created shot sequence video: {sequence_output_path}")
                else:
                    print(f"Failed to create shot sequence video")
                    
                # Also save the shot frame for reference
                shot_output_path = output_dir / f'made_shot_{shot_frame:05d}_{three_point_suffix}.png'
                create_visualization_with_court(
                    court_h5_path, 
                    player_h5_path, 
                    shot_frame, 
                    shooter_id, 
                    shot_output_path, 
                    False,
                    jersey_data,
                    None,
                    is_three_point,
                    three_point_analysis
                )
                
            else:
                print(f"Could not detect jump for shooter {shooter_id} (method: {detection_method})")
                # Save all tracked frames for debugging
                for frame_idx in all_frames:
                    frame_output_path = jump_detection_dir / f'tracked_frame_{frame_idx:05d}.png'
                    create_visualization_with_court(court_h5_path, player_h5_path, frame_idx, shooter_id, frame_output_path, jersey_data=jersey_data)
        else:
            print(f"Could not find shooter for shot at frame {shot_frame}")
            output_path = output_dir / f'shot_no_shooter_{shot_frame:05d}.png'
            create_visualization_with_court(court_h5_path, player_h5_path, shot_frame, None, output_path, jersey_data=jersey_data)

if __name__ == "__main__":
    main() 