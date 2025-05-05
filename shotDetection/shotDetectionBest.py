import numpy as np
import h5py
import cv2
from pathlib import Path
import os
import shutil
import json  # Added for JSON handling

def load_ball_data_from_csv(csv_path):
    """
    Load ball tracking data from CSV file.
    
    Args:
        csv_path: Path to the ball tracking CSV file
        
    Returns:
        Dictionary mapping frame numbers to ball data (center_x, center_y, radius, confidence, status)
    """
    ball_data = {}
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:  # Ensure we have all needed data
                    try:
                        frame_num = int(parts[0])
                        ball_data[frame_num] = {
                            'center_x': float(parts[1]),
                            'center_y': float(parts[2]),
                            'radius': float(parts[3]),
                            'confidence': float(parts[4]),
                            'status': parts[5]
                        }
                    except (ValueError, IndexError):
                        pass  # Skip malformed lines
    except Exception as e:
        print(f"Error loading ball data from CSV: {e}")
    
    return ball_data

def create_ball_mask(ball_data, frame_num, shape=(1080, 1920)):
    """
    Create a ball mask using CSV ball data for the specified frame.
    
    Args:
        ball_data: Dictionary of ball data (from load_ball_data_from_csv)
        frame_num: Frame number to create mask for
        shape: Shape of the output mask (height, width)
        
    Returns:
        Boolean mask of the ball, or None if no ball data for this frame
    """
    if frame_num not in ball_data:
        return None
    
    ball_info = ball_data[frame_num]
    
    # Skip empty/invalid ball data
    if ball_info['status'] == 'empty' or ball_info['radius'] <= 0:
        return None
    
    # Create empty mask
    mask = np.zeros(shape, dtype=bool)
    
    # Get ball center and radius in original 1920x1080 coordinates
    orig_center_x = float(ball_info['center_x'])
    orig_center_y = float(ball_info['center_y'])
    orig_radius = max(1, float(ball_info['radius']))
    
    # Scale coordinates to match the target shape
    orig_width, orig_height = 1920, 1080  # Original coordinate system
    scale_x = shape[1] / orig_width
    scale_y = shape[0] / orig_height
    
    center_x = int(orig_center_x * scale_x)
    center_y = int(orig_center_y * scale_y)
    radius = max(1, int(orig_radius * min(scale_x, scale_y)))  # Scale radius proportionally
    # Check if coordinates are within bounds of the frame
    if center_x < 0 or center_x >= shape[1] or center_y < 0 or center_y >= shape[0]:
        return None
    
    # Create the circular mask
    y_indices, x_indices = np.indices(shape)
    dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # Set pixels within radius to True
    mask[dist_from_center <= radius] = True
    
    # Debug print
    num_pixels = np.sum(mask)
    
    return mask

def identify_team(player_h5_path, video_path, frame_num, player_id):
    """
    Super simple team identification function.
    
    Args:
        player_h5_path: Path to the player tracking h5 file
        video_path: Path to the video file
        frame_num: Frame number to analyze
        player_id: ID of the player to identify team for
        
    Returns:
        team_id: 0 or 1 indicating which team the player belongs to
        team_name: String name of the team (e.g., "Blue Team")
    """
    import numpy as np
    import h5py
    import cv2
    
    # Step 1: Extract all player masks in this frame
    player_jerseys = {}  # Store player_id -> average jersey color
    
    # Open video and seek to frame
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        return 0, "Unknown Team"  # Default team if video can't be opened
    
    # Seek to the frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = video.read()
    video.release()
    
    if not success:
        return 0, "Unknown Team"  # Default team if frame can't be read
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Open H5 file and get player masks
    with h5py.File(player_h5_path, 'r') as h5_file:
        frame_key = f"frame_{frame_num:05d}"
        if frame_key not in h5_file:
            return 0, "Unknown Team"  # Default team if frame not in H5
        
        frame_group = h5_file[frame_key]
        
        # Process each player in this frame
        for obj_key in frame_group.keys():
            if not obj_key.startswith('player'):
                continue
                
            # Get player mask
            obj = frame_group[obj_key]
            if 'mask_rle' not in obj:
                continue
                
            rle_data = obj['mask_rle'][:]
            shape = tuple(obj['mask_shape'][:])
            
            # Decode RLE mask
            mask = np.zeros(shape[0] * shape[1], dtype=bool)
            for start, length in rle_data:
                mask[start:start+length] = True
            mask = mask.reshape(shape)
            
            # Resize mask if needed to match video dimensions
            if shape[0] != frame_height or shape[1] != frame_width:
                mask_resized = cv2.resize(mask.astype(np.uint8), (frame_width, frame_height)) > 0
            else:
                mask_resized = mask
            
            # Extract jersey region (middle 60% of the vertical axis)
            y_indices, x_indices = np.where(mask_resized)
            if len(y_indices) == 0:
                continue
                
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            height = max_y - min_y
            
            # Middle 60% of the mask (from 20% to 80% from top)
            jersey_top = min_y + int(height * 0.2)
            jersey_bottom = min_y + int(height * 0.8)
            
            # Create jersey mask
            jersey_mask = np.zeros_like(mask_resized)
            jersey_mask[jersey_top:jersey_bottom, :] = mask_resized[jersey_top:jersey_bottom, :]
            
            # Extract colors from the jersey region
            jersey_pixels = frame[jersey_mask]
            
            if len(jersey_pixels) > 10:  # Need enough pixels
                # Calculate average color
                avg_color = np.mean(jersey_pixels, axis=0)
                player_jerseys[obj_key] = avg_color
    
    # Check if we have at least 2 players
    if len(player_jerseys) < 2 or player_id not in player_jerseys:
        return 0, "Unknown Team"
    
    # Step 2: Calculate color distances between our player and everyone else
    target_color = player_jerseys[player_id]
    distances = {}
    
    for pid, color in player_jerseys.items():
        if pid != player_id:
            # Simple Euclidean distance in BGR space
            dist = np.sqrt(np.sum((target_color - color) ** 2))
            distances[pid] = dist
    
    # Step 3: Cluster into two teams (simple method - take median distance)
    distance_values = list(distances.values())
    median_distance = np.median(distance_values)
    
    # Count players with distance above/below median
    team_0_count = sum(1 for d in distance_values if d < median_distance)
    team_1_count = sum(1 for d in distance_values if d >= median_distance)
    
    # Determine team based on distances
    team_id = 1 if np.mean([d for pid, d in distances.items() if distances[pid] >= median_distance]) > median_distance else 0
    
    # Get representative team colors
    if team_id == 0:
        team_color = target_color
    else:
        # Average color of the other team
        other_team_colors = [color for pid, color in player_jerseys.items() 
                            if distances.get(pid, 0) >= median_distance]
        if other_team_colors:
            team_color = np.mean(other_team_colors, axis=0)
        else:
            team_color = np.array([255, 255, 255])  # Default white
    
    # Get team name based on color
    team_name = get_team_name(team_color)
    
    return team_id, team_name

def get_team_name(color):
    """
    Determine team name based on BGR color.
    
    Args:
        color: BGR color array
        
    Returns:
        team_name: String describing the team color
    """
    import cv2
    import numpy as np
    
    # Convert to HSV for better color analysis
    color_uint8 = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color_uint8, cv2.COLOR_BGR2HSV)[0][0]
    
    # Extract hue, saturation, value
    hue = hsv_color[0]
    saturation = hsv_color[1]
    value = hsv_color[2]
    
    # Simple color identification
    if saturation < 50 and value > 180:
        return "White Team"
    elif saturation < 50 and value < 70:
        return "Black Team"
    elif saturation < 50 and value < 180:
        return "Gray Team"
    elif 100 <= hue <= 140:
        return "Blue Team"
    elif 36 <= hue <= 80:
        return "Green Team"
    elif hue <= 20 or hue >= 160:
        return "Red Team"
    elif 20 < hue < 36:
        return "Orange Team"
    elif 80 < hue < 100:
        return "Teal Team"
    elif 140 < hue < 160:
        return "Purple Team"
    else:
        return f"Team (H:{hue}, S:{saturation}, V:{value})"

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
        
        if shots[i]['confidence'] >= 0.5:
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

def find_shooter_and_jump(h5_file_path, shot_frame, ball_data, max_lookback=200, distance_threshold=2, jump_detection_window=60):
    """
    Find the shooter and detect the jump frame with improved detection logic using CSV ball data.
    
    Args:
        h5_file_path: Path to the player tracking h5 file
        shot_frame: Frame where the shot was detected
        ball_data: Dictionary mapping frame numbers to ball data
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
            
            # Check if we have ball data for this frame
            if frame_idx not in ball_data:
                continue
            
            # Create ball mask from CSV data
            ball_info = ball_data[frame_idx]
            frame_shape = (1080, 1920)  # Default shape
            
            # Get first available player mask to determine actual frame shape
            for key in frame_group.keys():
                if key.startswith('player'):
                    player_mask = get_mask_from_player_h5(frame_group, key)
                    if player_mask is not None:
                        frame_shape = player_mask.shape
                        break
            
            # Create ball mask
            ball_mask = create_ball_mask(ball_data, frame_idx, frame_shape)
            if ball_mask is None or not np.any(ball_mask):
                continue
            
            # Get ball center coordinates
            ball_x = int(ball_info['center_x'])
            ball_y = int(ball_info['center_y'])
            
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
            
            # Check if we have ball data for this frame
            if frame_idx not in ball_data:
                continue
            
            # Get frame shape from first available player mask
            frame_shape = (1080, 1920)  # Default shape
            for key in frame_group.keys():
                if key.startswith('player'):
                    player_mask = get_mask_from_player_h5(frame_group, key)
                    if player_mask is not None:
                        frame_shape = player_mask.shape
                        break
            
            # Create ball mask
            ball_mask = create_ball_mask(ball_data, frame_idx, frame_shape)
            if ball_mask is None or not np.any(ball_mask):
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
    
    # Create a sorted list of players by average distance
    if not avg_distances:
        return None, None, None, [], None, "no_shooter_found"
        
    sorted_players = sorted(avg_distances.items(), key=lambda x: x[1])
    
    # Try each player in order of increasing distance until we find one that passes continuity check
    for player_entry in sorted_players:
        player_key = player_entry[0]  # The player ID
        player_avg_distance = player_entry[1]  # The average distance
        
        # Get object_id for this player
        object_id = player_object_ids.get(player_key)
        
        # *** CONTINUITY CHECK ***
        # Verify that the detected player exists in at least one frame before and after the detection frame
        player_exists_continuity = False
        detection_frame = last_touch_frame  # The frame where the shooter was initially detected
        
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Check frames before detection
            found_before = False
            for frame_idx in range(detection_frame - 1, max(0, detection_frame - 10), -1):
                frame_key = f"frame_{frame_idx:05d}"
                if frame_key not in h5_file:
                    continue
                    
                frame_group = h5_file[frame_key]
                
                if player_key in frame_group:
                    # Also verify player has the same object_id for continuity
                    player_obj = frame_group[player_key]
                    player_attrs = dict(player_obj.attrs)
                    current_object_id = player_attrs.get('object_id', None)
                    
                    # Only count as valid if object_id matches or both are None
                    if current_object_id == object_id or (current_object_id is None and object_id is None):
                        found_before = True
                        break
            
            # Check frames after detection
            found_after = False
            for frame_idx in range(detection_frame + 1, min(shot_frame + 1, detection_frame + 10)):
                frame_key = f"frame_{frame_idx:05d}"
                if frame_key not in h5_file:
                    continue
                    
                frame_group = h5_file[frame_key]
                
                if player_key in frame_group:
                    # Also verify player has the same object_id for continuity
                    player_obj = frame_group[player_key]
                    player_attrs = dict(player_obj.attrs)
                    current_object_id = player_attrs.get('object_id', None)
                    
                    # Only count as valid if object_id matches or both are None
                    if current_object_id == object_id or (current_object_id is None and object_id is None):
                        found_after = True
                        break
            
            player_exists_continuity = found_before and found_after
        
        # If player doesn't pass continuity check, try the next candidate
        if not player_exists_continuity:
            print(f"Player {player_key} (object_id: {object_id}) failed continuity check, trying next candidate")
            continue
        
        # If we get here, we found a valid player - select it and break
        closest_player_overall = player_key
        object_id_overall = object_id
        detection_method = "temporal_average_distance"
        break
    
    # If no player passed the continuity check, return no shooter
    if closest_player_overall is None:
        return None, None, None, [], None, "all_candidates_failed_continuity"
    
    # Now collect data for the shooter and ball across all relevant frames
    jump_frames = []
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        for frame_idx in range(shot_frame, max(0, shot_frame - jump_detection_window), -1):
            frame_key = f"frame_{frame_idx:05d}"
            if frame_key not in h5_file:
                continue
                
            frame_group = h5_file[frame_key]
            
            # We need both player and ball to be present
            if closest_player_overall not in frame_group or frame_idx not in ball_data:
                continue
                
            # Get player mask and position
            player_mask = get_mask_from_player_h5(frame_group, closest_player_overall)
            if player_mask is None or not np.any(player_mask):
                continue
                
            # Get frame shape
            frame_shape = player_mask.shape
            
            # Create ball mask
            ball_mask = create_ball_mask(ball_data, frame_idx, frame_shape)
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

def determine_court_orientation(keyway_mask, three_point_mask):
    """
    Determine if the court view is looking left or right by analyzing the keyway and three-point line.
    
    Args:
        keyway_mask: Binary mask of the keyway (free throw lane)
        three_point_mask: Binary mask of the three-point line
        
    Returns:
        orientation: String "looking_right" or "looking_left"
    """
    # Get keyway and three-point line coordinates
    keyway_y, keyway_x = np.where(keyway_mask)
    three_pt_y, three_pt_x = np.where(three_point_mask)
    
    if len(keyway_y) == 0 or len(three_pt_y) == 0:
        return "unknown"
    
    # Get keyway centroid
    keyway_center_x = np.mean(keyway_x)
    keyway_center_y = np.mean(keyway_y)
    
    # Create horizontal scan from keyway center
    center_y_idx = int(keyway_center_y)
    left_scan = three_point_mask[center_y_idx, :int(keyway_center_x)]
    right_scan = three_point_mask[center_y_idx, int(keyway_center_x):]
    
    # Check which direction has three-point line intersection
    left_intersects = np.any(left_scan)
    right_intersects = np.any(right_scan)
    
    # Determine court orientation
    if left_intersects and not right_intersects:
        return "looking_right"  # Extending left hits three-point line
    elif right_intersects and not left_intersects:
        return "looking_left"   # Extending right hits three-point line
    elif left_intersects and right_intersects:
        # Both sides intersect, check which is closer
        left_distances = np.where(left_scan)[0]
        right_distances = np.where(right_scan)[0]
        
        left_distance = keyway_center_x - np.max(left_distances) if len(left_distances) > 0 else float('inf')
        right_distance = np.min(right_distances) if len(right_distances) > 0 else float('inf')
        
        return "looking_right" if left_distance < right_distance else "looking_left"
    else:
        # Fallback to centroid comparison method
        three_pt_center_x = np.mean(three_pt_x)
        if keyway_center_x < three_pt_center_x:
            return "looking_right"  # Basket is on the right side
        else:
            return "looking_left"   # Basket is on the left side


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
    
    # ----- Determine court orientation using the dedicated function -----
    court_orientation = determine_court_orientation(keyway_mask, three_point_mask)
    three_point_analysis["court_orientation"] = court_orientation
    
    # ----- Find closest point on three-point line to player's feet -----
    player_y, player_x = player_foot_position
    min_distance = float('inf')
    closest_point = None
    
    # To avoid excessive computation, sample points from the three-point line
    three_pt_y, three_pt_x = np.where(three_point_mask)
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
        
        # Get keyway center for vector calculation
        keyway_y, keyway_x = np.where(keyway_mask)
        keyway_center_y = np.mean(keyway_y)
        keyway_center_x = np.mean(keyway_x)
        
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

def detect_rim_position(court_h5_path, player_h5_path, frame_num):
    """
    Detect the rim position in a frame by looking at court markings.
    
    Args:
        court_h5_path: Path to the court tracking h5 file
        player_h5_path: Path to the player tracking h5 file
        frame_num: Frame number to analyze
        
    Returns:
        rim_position: (y, x) coordinates of the rim center
        rim_found: Boolean indicating if the rim was successfully detected
    """
    import numpy as np
    import h5py
    import cv2
    
    # Default return values
    rim_position = None
    rim_found = False
    
    try:
        with h5py.File(court_h5_path, 'r') as court_h5:
            # Find the closest court frame
            court_frame_key = find_closest_court_frame(court_h5_path, frame_num)
            
            frames_group = court_h5
            if 'frames' in court_h5:
                frames_group = court_h5['frames']
            
            if court_frame_key is not None and court_frame_key in frames_group:
                frame = frames_group[court_frame_key]
                
                # Get keyway (free throw lane) mask - usually class_id 1
                keyway_mask = None
                for detection_key in frame.keys():
                    detection = frame[detection_key]
                    
                    class_id = detection.attrs.get('class_id', detection.attrs.get('class', 0))
                    
                    if class_id == 1:  # Keyway
                        rle_data = detection['rle'][:]
                        mask_shape = tuple(detection['rle'].attrs['shape'])
                        keyway_mask = rle_decode(rle_data, mask_shape)
                        break
                
                if keyway_mask is None:
                    return None, False
                
                # Get dimensions from any existing mask
                frame_shape = None
                for detection_key in frame.keys():
                    detection = frame[detection_key]
                    rle_data = detection['rle'][:]
                    mask_shape = tuple(detection['rle'].attrs['shape'])
                    frame_shape = mask_shape
                    break
                
                if frame_shape is None:
                    return None, False
                
                # Get court orientation
                keyway_y, keyway_x = np.where(keyway_mask)
                
                if len(keyway_y) == 0:
                    return None, False
                
                # Calculate keyway endpoints
                min_keyway_x = np.min(keyway_x)
                max_keyway_x = np.max(keyway_x)
                min_keyway_y = np.min(keyway_y)
                max_keyway_y = np.max(keyway_y)
                
                # Get court orientation
                court_orientation = "unknown"
                keyway_center_x = (min_keyway_x + max_keyway_x) // 2
                keyway_center_y = (min_keyway_y + max_keyway_y) // 2
                
                # Check which side of the frame has the highest concentration of keyway pixels
                left_half_count = np.sum(keyway_x < frame_shape[1] // 2)
                right_half_count = np.sum(keyway_x >= frame_shape[1] // 2)
                
                if left_half_count > right_half_count:
                    court_orientation = "looking_right"  # Basket is on the left side
                    # The rim should be at the left edge of the keyway
                    rim_x = min_keyway_x
                else:
                    court_orientation = "looking_left"   # Basket is on the right side
                    # The rim should be at the right edge of the keyway
                    rim_x = max_keyway_x
                
                # Estimate rim y-position (usually at the bottom edge of the keyway)
                # We'll use the vertical position that has the most keyway pixels near the rim_x
                rim_y_candidates = []
                
                # Look at pixels within a small window around the estimated rim_x
                window_size = 20
                
                for y in range(min_keyway_y, max_keyway_y + 1):
                    # Count keyway pixels in this row near rim_x
                    count = 0
                    for x in range(max(0, rim_x - window_size), min(frame_shape[1], rim_x + window_size)):
                        if keyway_mask[y, x]:
                            count += 1
                    
                    if count > 0:
                        rim_y_candidates.append((y, count))
                
                if rim_y_candidates:
                    # Get the y with the highest count
                    rim_y_candidates.sort(key=lambda x: x[1], reverse=True)
                    rim_y = rim_y_candidates[0][0]
                else:
                    # Fallback: use the keyway center y
                    rim_y = keyway_center_y
                
                # Return the estimated rim position
                rim_position = (rim_y, rim_x)
                rim_found = True
    
    except Exception as e:
        print(f"Error detecting rim position: {e}")
        return None, False
    
    return rim_position, rim_found

def verify_ball_above_rim(player_h5_path, shot_frame, ball_data, rim_position=None, lookback_frames=8):
    """
    Verify if the ball was above the rim in the frames leading up to the shot using CSV ball data.
    
    Args:
        player_h5_path: Path to the player tracking h5 file
        shot_frame: Frame where the shot was detected
        ball_data: Dictionary of ball data from CSV
        rim_position: (y, x) coordinates of the rim, or None to attempt automatic detection
        lookback_frames: Number of frames to look back from the shot frame
        
    Returns:
        is_valid_shot: Boolean indicating if the ball was above the rim
        max_height_frame: Frame where the ball reached its highest point
        max_height_diff: Maximum height difference between ball and rim (negative means ball above rim)
    """
    # Default return values
    is_valid_shot = False
    max_height_frame = None
    max_height_diff = float('inf')
    
    # If no rim position is provided, we can't verify
    if rim_position is None:
        return False, None, float('inf')
    
    rim_y, rim_x = rim_position
    
    try:
        # Examine each frame within the lookback window
        for frame_idx in range(max(0, shot_frame - lookback_frames), shot_frame + 1):
            # Check if we have ball data for this frame
            if frame_idx not in ball_data:
                continue
            
            # Get ball position directly from CSV data
            ball_info = ball_data[frame_idx]
            ball_y = int(ball_info['center_y'])
            
            # Calculate height difference between ball and rim
            # Negative value means ball is above the rim
            # Need to account for ball radius to check if the bottom of the ball is above rim
            ball_radius = max(1, int(ball_info['radius']))
            ball_bottom_y = ball_y + ball_radius
            height_diff = ball_bottom_y - rim_y
            
            # Update max height if this is the highest point
            if height_diff < max_height_diff:
                max_height_diff = height_diff
                max_height_frame = frame_idx
            
            # If ball is above the rim at any point, mark as valid shot
            if height_diff < 0:
                is_valid_shot = True
    
    except Exception as e:
        print(f"Error verifying ball above rim: {e}")
        return False, None, float('inf')
    
    return is_valid_shot, max_height_frame, max_height_diff

def check_shot_validity(court_h5_path, player_h5_path, shot_frame, ball_data, lookback_frames=8):
    """
    Check if a shot is valid by verifying if the ball was above the rim using CSV ball data.
    
    Args:
        court_h5_path: Path to the court tracking h5 file
        player_h5_path: Path to the player tracking h5 file
        shot_frame: Frame where the shot was detected
        ball_data: Dictionary mapping frame numbers to ball data from CSV
        lookback_frames: Number of frames to look back from the shot frame
    
    Returns:
        is_valid_shot: Boolean indicating if the shot is valid
        validation_data: Dictionary with details of the validation
    """
    # First, detect the rim position
    rim_position, rim_found = detect_rim_position(court_h5_path, player_h5_path, shot_frame)
    
    validation_data = {
        "rim_position": rim_position,
        "rim_found": rim_found,
        "max_height_frame": None,
        "max_height_diff": float('inf'),
        "failure_reason": None
    }
    
    if not rim_found:
        validation_data["failure_reason"] = "rim_not_detected"
        return False, validation_data
    
    # Then check if the ball was above the rim using CSV ball data
    is_valid_shot, max_height_frame, max_height_diff = verify_ball_above_rim(
        player_h5_path, shot_frame, ball_data, rim_position, lookback_frames
    )
    
    validation_data["max_height_frame"] = max_height_frame
    validation_data["max_height_diff"] = max_height_diff
    
    if not is_valid_shot:
        validation_data["failure_reason"] = "ball_not_above_rim"
    
    return is_valid_shot, validation_data

def create_visualization_with_court(court_h5_path, player_h5_path, frame_num, shooter_id, output_path, 
                                   is_jump_frame=False, jersey_data=None, detection_method=None,
                                   is_three_point=None, three_point_analysis=None, shooter_frame=None,
                                   shooter_detection_method=None, shooter_ball_distance=None, ball_data=None):
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
    
    # First draw ball from CSV data (underneath players)
    if ball_data is not None and frame_num in ball_data:
        ball_info = ball_data[frame_num]
        ball_x = int(ball_info['center_x'])
        ball_y = int(ball_info['center_y'])
        ball_radius = max(5, int(ball_info['radius']))  # Ensure minimum visible radius
        
        # Draw the ball as an orange circle
        cv2.circle(combined_canvas, (ball_x, ball_y), ball_radius, (0, 165, 255), -1)  # Filled orange circle
    
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
                    if obj_key == 'ball':  # Skip the ball from H5
                        continue
                        
                    mask = get_mask_from_player_h5(frame_group, obj_key)
                    if mask is None:
                        continue
                    
                    # Get object_id if this is the shooter
                    if obj_key == shooter_id:
                        player_obj = frame_group[obj_key]
                        player_attrs = dict(player_obj.attrs)
                        shooter_object_id = player_attrs.get('object_id', None)
                    
                    # Determine coloring:
                    # - Yellow for shooter at jump frame
                    # - Green for shooter in other frames after jump
                    # - White for non-shooters
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
                            
                            # Highlight the shooter detection frame
                            if shooter_frame is not None and frame_num == shooter_frame:
                                # Apply a special highlight for shooter detection frame
                                if bottom_mask_20_percent.shape[0] != combined_canvas.shape[0] or bottom_mask_20_percent.shape[1] != combined_canvas.shape[1]:
                                    temp_canvas = np.zeros((bottom_mask_20_percent.shape[0], bottom_mask_20_percent.shape[1], 3), dtype=np.uint8)
                                    temp_canvas[bottom_mask_20_percent] = (0, 0, 255)  # Red for shooter detection frame
                                    
                                    temp_canvas = cv2.resize(temp_canvas, (combined_canvas.shape[1], combined_canvas.shape[0]))
                                    
                                    resized_mask = np.any(temp_canvas > 0, axis=2)
                                    for c in range(3):
                                        # Apply a semi-transparent overlay (75% opacity)
                                        combined_canvas[:, :, c][resized_mask] = combined_canvas[:, :, c][resized_mask] * 0.25 + temp_canvas[:, :, c][resized_mask] * 0.75
                                else:
                                    for c in range(3):
                                        # Apply a semi-transparent overlay (75% opacity)
                                        combined_canvas[:, :, c][bottom_mask_20_percent] = combined_canvas[:, :, c][bottom_mask_20_percent] * 0.25 + (255 if c == 2 else 0) * 0.75
                                
                                # Add text explaining this is where shooter was detected
                                cv2.putText(combined_canvas, f"SHOOTER DETECTED HERE", 
                                           (avg_x - 100, avg_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (0, 0, 0), 2)
                                cv2.putText(combined_canvas, f"SHOOTER DETECTED HERE", 
                                           (avg_x - 100, avg_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (255, 255, 255), 1)
                                
                                # If we have ball information, draw connection to ball
                                if ball_data is not None and frame_num in ball_data:
                                    ball_info = ball_data[frame_num]
                                    ball_center_x = int(ball_info['center_x'])
                                    ball_center_y = int(ball_info['center_y'])
                                    # Draw a line from player to ball
                                    cv2.line(combined_canvas, (avg_x, avg_y), (ball_center_x, ball_center_y), 
                                            (0, 0, 255), 2)
                            
                            # For jump frames, apply enhanced visualization
                            elif is_jump_frame:
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
    
    except Exception as e:
        print(f"Error processing player data: {e}")
    
    # Draw the ball again on top to ensure visibility (after players)
    if ball_data is not None and frame_num in ball_data:
        ball_info = ball_data[frame_num]
        ball_x = int(ball_info['center_x'])
        ball_y = int(ball_info['center_y'])
        ball_radius = max(5, int(ball_info['radius']))  # Ensure minimum visible radius
        
        # Draw circle outline for contrast
        cv2.circle(combined_canvas, (ball_x, ball_y), ball_radius+2, (0, 0, 0), 2)  # Black outline
        # Add orange ball if it was obscured
        if np.sum(combined_canvas[ball_y, ball_x]) > 0 and np.sum(combined_canvas[ball_y, ball_x]) != np.sum((0, 165, 255)):
            cv2.circle(combined_canvas, (ball_x, ball_y), ball_radius, (0, 165, 255), -1)  # Filled orange circle
    
    # Add frame number and phase indication
    frame_text = f"Frame: {frame_num}"
    if shooter_frame is not None and frame_num == shooter_frame:
        frame_text += f" (SHOOTER DETECTION FRAME - {shooter_detection_method})"
        if shooter_ball_distance is not None:
            frame_text += f" - Dist: {shooter_ball_distance:.2f}px"
    elif is_jump_frame:
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

def create_jump_sequence_video(court_h5_path, player_h5_path, jump_frame, shot_frame, shooter_id, output_path, 
                               ball_data, fps=10, pre_jump_frames=20, jersey_data=None, detection_method=None, 
                               shooter_frame=None, shooter_detection_method=None, shooter_ball_distance=None):
    """
    Create a video of the shot sequence from pre-jump to shot frame using CSV ball data.
    
    Args:
        court_h5_path: Path to the court tracking h5 file
        player_h5_path: Path to the player tracking h5 file
        jump_frame: Frame where the jump begins
        shot_frame: Frame where the shot was detected
        shooter_id: ID of the shooter player
        output_path: Path to save the output video
        ball_data: Dictionary mapping frame numbers to ball data from CSV
        fps: Frames per second for the output video
        pre_jump_frames: Number of frames to include before the jump
        jersey_data: Dictionary with jersey number information
        detection_method: Method used to detect the jump
        shooter_frame: Frame where the shooter was identified
        shooter_detection_method: Method used to identify the shooter
        shooter_ball_distance: Distance between shooter and ball at identification
        
    Returns:
        success: Boolean indicating if the video was created successfully
        team_type: String indicating if this is a home or away team
    """
    # If shooter_frame is provided, use it to determine starting point
    # Otherwise, just use the jump_frame minus pre_jump_frames
    if shooter_frame is not None:
        # Start from shooter frame and go back additional 20 frames for context
        start_frame = max(1, shooter_frame - 20)
    else:
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
        
        # Determine if this is home or away based on court orientation
        court_orientation = three_point_analysis.get("court_orientation", "unknown")
        team_type = "home" if court_orientation == "looking_left" else "away"
        
        print(f"Court orientation: {court_orientation} - Classified as: {team_type}")
        
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
                current_three_point_analysis,
                shooter_frame,
                shooter_detection_method,
                shooter_ball_distance,
                ball_data  # Pass the ball data to the visualization function
            )
        
        # Check if we have any frames
        if not frame_paths:
            print("No frames were generated for the video.")
            return False, None
        
        # Get frame dimensions from the first frame
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            print(f"Could not read frame: {frame_paths[0]}")
            return False, None
            
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
        
        return True, team_type
        
    except Exception as e:
        print(f"Error creating sequence video: {e}")
        return False, None
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

def find_shooter_and_jump_with_team_verification(h5_file_path, shot_frame, court_h5_path, video_path, 
                                                ball_data, expected_team_looking_left, expected_team_looking_right, 
                                                max_lookback=200, distance_threshold=2, jump_detection_window=60):
    """
    Find the shooter and detect the jump with team verification using CSV ball data.
    
    Args:
        h5_file_path: Path to the player tracking h5 file
        shot_frame: Frame where the shot was detected
        court_h5_path: Path to the court tracking h5 file
        video_path: Path to the video file
        ball_data: Dictionary mapping frame numbers to ball data from CSV
        expected_team_looking_left: Expected team color for the team looking left
        expected_team_looking_right: Expected team color for the team looking right
        max_lookback: Maximum frames to look back
        distance_threshold: Maximum distance between ball and player
        jump_detection_window: Maximum frames to look back for jump detection
        
    Returns:
        Various detection data including shooter_id, jump_frame, etc.
    """
    # First, use the updated function to find the shooter and jump with CSV ball data
    shooter_id, jump_frame, jump_position, jump_frames, object_id, detection_method = find_shooter_and_jump(
        h5_file_path, shot_frame, ball_data, max_lookback, distance_threshold, jump_detection_window
    )
    
    is_team_match = False
    shooter_frame = None
    shooter_detection_method = "unknown"
    shooter_ball_distance = float('inf')
    
    # If no shooter was found, return early with team_match as False
    if shooter_id is None:
        return shooter_id, jump_frame, jump_position, jump_frames, object_id, detection_method, is_team_match, shooter_frame, shooter_detection_method, shooter_ball_distance
    
    # If shooter was found but jump wasn't, we still have a shooter_frame
    # Get shooter frame - this is the frame where we first identified this player as the shooter
    shooter_frame = shot_frame
    with h5py.File(h5_file_path, 'r') as h5_file:
        # First check if we can find the last touch frame (ball close to player)
        for frame_idx in range(shot_frame, max(0, shot_frame - max_lookback), -1):
            frame_key = f"frame_{frame_idx:05d}"
            if frame_key not in h5_file or frame_idx not in ball_data:
                continue
                
            frame_group = h5_file[frame_key]
            
            if shooter_id not in frame_group:
                continue
            
            # Get player mask
            player_mask = get_mask_from_player_h5(frame_group, shooter_id)
            if player_mask is None or not np.any(player_mask):
                continue
            
            # Create ball mask from CSV data
            frame_shape = player_mask.shape
            ball_mask = create_ball_mask(ball_data, frame_idx, frame_shape)
            if ball_mask is None or not np.any(ball_mask):
                continue
                
            # Ball and player are both present, calculate minimum distance
            ball_pixels = np.where(ball_mask)
            player_pixels = np.where(player_mask)
            
            if len(ball_pixels[0]) == 0 or len(player_pixels[0]) == 0:
                continue
                
            ball_points = np.column_stack((ball_pixels[0], ball_pixels[1]))
            player_points = np.column_stack((player_pixels[0], player_pixels[1]))
            
            distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
            min_distance = np.min(distances)
            
            # If distance is small enough, this is where we identified the shooter
            if min_distance <= distance_threshold:
                shooter_frame = frame_idx
                shooter_ball_distance = min_distance
                shooter_detection_method = "last_touch_point"
                break
        
        # If we couldn't find a frame with distance below the threshold, try to find
        # the first frame where the player and ball are both present
        if shooter_detection_method == "unknown":
            # Check if we can identify a "temporal window" frame
            for frame_idx in range(shot_frame, max(0, shot_frame - max_lookback), -1):
                frame_key = f"frame_{frame_idx:05d}"
                if frame_key not in h5_file or frame_idx not in ball_data:
                    continue
                    
                frame_group = h5_file[frame_key]
                
                if shooter_id not in frame_group:
                    continue
                
                # Get player mask
                player_mask = get_mask_from_player_h5(frame_group, shooter_id)
                if player_mask is None or not np.any(player_mask):
                    continue
                
                # Create ball mask
                frame_shape = player_mask.shape
                ball_mask = create_ball_mask(ball_data, frame_idx, frame_shape)
                if ball_mask is None or not np.any(ball_mask):
                    continue
                    
                # Calculate distance anyway for reporting
                ball_pixels = np.where(ball_mask)
                player_pixels = np.where(player_mask)
                
                if len(ball_pixels[0]) > 0 and len(player_pixels[0]) > 0:
                    ball_points = np.column_stack((ball_pixels[0], ball_pixels[1]))
                    player_points = np.column_stack((player_pixels[0], player_pixels[1]))
                    
                    distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
                    min_distance = np.min(distances)
                    
                    shooter_frame = frame_idx
                    shooter_ball_distance = min_distance
                    shooter_detection_method = "temporal_window_analysis"
                    break
    
    # Determine the court orientation - looking left or looking right
    court_orientation = "unknown"
    
    # Use the three point analysis to determine court orientation
    three_point_check = check_three_point_shot(court_h5_path, h5_file_path, jump_frame if jump_frame is not None else shooter_frame, shooter_id)
    if three_point_check and len(three_point_check) >= 3:
        _, _, three_point_analysis = three_point_check
        court_orientation = three_point_analysis.get("court_orientation", "unknown")
    
    # If court orientation is unknown, try using a direct court orientation check
    if court_orientation == "unknown":
        try:
            with h5py.File(court_h5_path, 'r') as court_h5:
                court_frame_key = find_closest_court_frame(court_h5_path, jump_frame if jump_frame is not None else shooter_frame)
                
                frames_group = court_h5
                if 'frames' in court_h5:
                    frames_group = court_h5['frames']
                
                if court_frame_key is not None and court_frame_key in frames_group:
                    frame = frames_group[court_frame_key]
                    
                    # Get keyway and three-point line masks for orientation detection
                    keyway_mask = None
                    three_point_mask = None
                    
                    for detection_key in frame.keys():
                        detection = frame[detection_key]
                        class_id = detection.attrs.get('class_id', detection.attrs.get('class', 0))
                        
                        rle_data = detection['rle'][:]
                        mask_shape = tuple(detection['rle'].attrs['shape'])
                        mask = rle_decode(rle_data, mask_shape)
                        
                        if class_id == 1:  # Keyway
                            keyway_mask = mask
                        elif class_id == 2:  # Three-point line
                            three_point_mask = mask
                    
                    if keyway_mask is not None and three_point_mask is not None:
                        court_orientation = determine_court_orientation(keyway_mask, three_point_mask)
        except Exception as e:
            print(f"Error determining court orientation: {e}")
    
    # Check what team the shooter is on
    team_id, team_name = identify_team(h5_file_path, video_path, jump_frame if jump_frame is not None else shooter_frame, shooter_id)
    team_color = team_name.split()[0]  # Extract color from team name (e.g., "Blue" from "Blue Team")
    
    # Compare team with expected team based on court orientation
    if court_orientation == "looking_left" and team_color == expected_team_looking_left:
        is_team_match = True
    elif court_orientation == "looking_right" and team_color == expected_team_looking_right:
        is_team_match = True
    
    return shooter_id, jump_frame, jump_position, jump_frames, object_id, detection_method, is_team_match, shooter_frame, shooter_detection_method, shooter_ball_distance


def main():
    fileName = "2ndQIp"
    court_h5_path = f'../courtTrackingOutputs/{fileName}.h5'
    player_h5_path = f'../trackingOutputs/{fileName}.h5'
    shot_csv_path = f'../madeShotCSV/{fileName}.csv'
    jersey_json_path = f'../jerseyNumbers/{fileName}_jersey.json'
    video_path = f'../clips/{fileName}.mp4'
    ball_csv_path = f'../ballData/{fileName}_ball.csv'  # New path for ball CSV data
    
    # Load ball data from CSV
    ball_data = load_ball_data_from_csv(ball_csv_path)
    print(f"Loaded ball data for {len(ball_data)} frames from {ball_csv_path}")
    
    # Define distance threshold for shooter detection
    distance_threshold = 2  # Same as default in find_shooter_and_jump_with_team_verification function
    
    # Define expected team colors for each direction
    expected_team_looking_left = "Gray"  
    expected_team_looking_right = "Blue"
    
    # Create output directories (removed jumpDetection directory)
    output_dir = Path('madeShots')
    output_dir.mkdir(exist_ok=True)
    
    sequences_dir = Path('shotSequences')
    sequences_dir.mkdir(exist_ok=True)
    
    # Create a new directory for invalid shot sequences
    invalid_sequences_dir = Path('shotSequences/invalid')
    invalid_sequences_dir.mkdir(exist_ok=True, parents=True)
    
    detection_logs_dir = Path('detectionLogs')
    detection_logs_dir.mkdir(exist_ok=True)
    
    three_point_dir = Path('threePointAnalysis')
    three_point_dir.mkdir(exist_ok=True)
    
    # Create a new directory for shot validation
    shot_validation_dir = Path('shotValidation')
    shot_validation_dir.mkdir(exist_ok=True)
    
    # Create a directory for team verification
    team_verification_dir = Path('teamVerification')
    team_verification_dir.mkdir(exist_ok=True)
    
    # Create a log file for detection methods with 3-point information and team info
    log_file = detection_logs_dir / f'{fileName}_detection_methods.csv'
    with open(log_file, 'w') as f:
        f.write("shot_frame,jump_frame,shooter_id,detection_method,jersey_number,is_three_point,distance_to_line,team,is_valid_shot,validation_reason,is_team_match,court_orientation,shooter_frame,shooter_detection_method,shooter_ball_distance\n")
    
    # Create a scoring tally to track points
    scoring_tally = {
        "home": {
            "2PT": 0,
            "3PT": 0,
            "total": 0
        },
        "away": {
            "2PT": 0,
            "3PT": 0,
            "total": 0
        },
        "players": {}  # Will store by player jersey number and team
    }
    
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
        
        # First, check if the shot is valid (ball above rim in previous 8 frames)
        is_valid_shot, validation_data = check_shot_validity(court_h5_path, player_h5_path, shot_frame, ball_data, lookback_frames=8)
        validity_suffix = "valid" if is_valid_shot else "invalid"
        
        if is_valid_shot:
            print(f"VALID SHOT: Ball was above rim at frame {validation_data['max_height_frame']}")
            print(f"Maximum height difference: {validation_data['max_height_diff']:.2f} pixels")
        else:
            print(f"INVALID SHOT: {validation_data['failure_reason']}")
            
            # Create a visualization for invalid shots to help debug
            invalid_viz_path = shot_validation_dir / f'invalid_shot_{shot_frame:05d}_{validation_data["failure_reason"]}.png'
            create_visualization_with_court(court_h5_path, player_h5_path, shot_frame, None, invalid_viz_path, False, None, None, None, None, None, None, None, ball_data)
            
            # If the rim was found but ball wasn't above it, visualize the max height frame
            if validation_data['rim_found'] and validation_data['max_height_frame'] is not None:
                max_height_viz_path = shot_validation_dir / f'max_height_{validation_data["max_height_frame"]:05d}.png'
                create_visualization_with_court(court_h5_path, player_h5_path, validation_data['max_height_frame'], None, max_height_viz_path, False, None, None, None, None, None, None, None, ball_data)
        
        # Find shooter and detect jump - now with team verification and enhanced shooter detection info
        shooter_id, jump_frame, jump_position, all_frames, object_id, detection_method, is_team_match, shooter_frame, shooter_detection_method, shooter_ball_distance = find_shooter_and_jump_with_team_verification(
            player_h5_path, 
            shot_frame,
            court_h5_path,
            video_path,
            ball_data,  # Pass ball data to the function
            expected_team_looking_left,
            expected_team_looking_right,
            distance_threshold=distance_threshold  # Pass the defined distance threshold explicitly
        )
        
        # Determine court orientation for logging
        court_orientation = "unknown"
        if shooter_id is not None and jump_frame is not None:
            three_point_check = check_three_point_shot(court_h5_path, player_h5_path, jump_frame, shooter_id)
            if three_point_check and len(three_point_check) >= 3:
                _, _, three_point_analysis = three_point_check
                court_orientation = three_point_analysis.get("court_orientation", "unknown")
        
        # Get jersey number if available
        jersey_number = "unknown"
        team_name = "Unknown Team"
        
        if shooter_id is not None:
            # Print ENHANCED shooter detection information
            print(f"Found shooter {shooter_id} at frame {shooter_frame}")
            print(f"SHOOTER DETECTION METHOD: {shooter_detection_method}")
            print(f"SHOOTER BALL DISTANCE: {shooter_ball_distance:.2f} pixels (threshold: {distance_threshold} pixels)")
            
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
            
            # Identify shooter's team
            team_id, team_name = identify_team(player_h5_path, video_path, shot_frame, shooter_id)
            print(f"SHOOTER TEAM: {team_name} (Team ID: {team_id})")
            
            # Add team verification status
            print(f"TEAM MATCH: {'Yes' if is_team_match else 'No'}")
            
            # Create visualization for team verification
            if jump_frame is not None:
                team_viz_path = team_verification_dir / f'team_verification_{jump_frame:05d}_{team_name}_{court_orientation}_match-{is_team_match}.png'
                create_visualization_with_court(court_h5_path, player_h5_path, jump_frame, shooter_id, team_viz_path, True, jersey_data, None, None, None, None, None, None, ball_data)
        
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
                
                # Update court orientation if not already set
                if court_orientation == "unknown" and three_point_analysis:
                    court_orientation = three_point_analysis.get("court_orientation", "unknown")
                
                print(f"Three-point shot analysis:")
                print(f"  Is three-point shot: {is_three_point}")
                print(f"  Distance to 3PT line: {proximity_to_line:.2f} pixels")
                print(f"  Analysis details: {three_point_analysis}")
                print(f"  Court orientation: {court_orientation}")
                
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
                    three_point_analysis,
                    shooter_frame,
                    shooter_detection_method,
                    shooter_ball_distance,
                    ball_data  # Pass ball data to the function
                )
            except Exception as e:
                # If there's an error in three-point analysis, log it but continue processing
                print(f"Error analyzing three-point shot: {e}")
                is_three_point = False
                proximity_to_line = float('inf')
                three_point_analysis = None
        
        # Determine team_type (home or away)
        team_type = "unknown"
        if court_orientation == "looking_left":
            team_type = "home"
        elif court_orientation == "looking_right":
            team_type = "away"
        
        # Update scoring tally if the shot is valid
        if is_valid_shot and shooter_id is not None:
            points = 3 if is_three_point else 2
            point_type = "3PT" if is_three_point else "2PT"
            
            # Update team total
            if team_type in scoring_tally:
                scoring_tally[team_type][point_type] += 1
                scoring_tally[team_type]["total"] += points
            
            # Update player total
            player_key = f"{team_type}_{jersey_number}" if jersey_number != "unknown" else f"{team_type}_undetected"
            
            if player_key not in scoring_tally["players"]:
                scoring_tally["players"][player_key] = {
                    "2PT": 0,
                    "3PT": 0,
                    "total": 0
                }
            
            scoring_tally["players"][player_key][point_type] += 1
            scoring_tally["players"][player_key]["total"] += points
        
        # Log the detection information with 3-point data, team info, and shot validation
        with open(log_file, 'a') as f:
            three_point_str = "Yes" if is_three_point else "No"
            distance_str = f"{proximity_to_line:.2f}" if proximity_to_line != float('inf') else "N/A"
            valid_shot_str = "Valid" if is_valid_shot else "Invalid"
            validation_reason = "ball_above_rim" if is_valid_shot else validation_data.get('failure_reason', 'unknown')
            team_match_str = "Yes" if is_team_match else "No"
            shooter_frame_str = str(shooter_frame) if shooter_frame is not None else "None"
            
            f.write(f"{shot_frame},{jump_frame if jump_frame else 'None'},{shooter_id if shooter_id else 'None'},{detection_method},"
                   f"{jersey_number},{three_point_str},{distance_str},{team_name},{valid_shot_str},{validation_reason},"
                   f"{team_match_str},{court_orientation},{shooter_frame_str},{shooter_detection_method},{shooter_ball_distance:.2f}\n")
        
        # Create a detailed log for this shot
        shot_log_file = detection_logs_dir / f'shot_{shot_frame}_details.txt'
        with open(shot_log_file, 'w') as f:
            f.write(f"Shot Frame: {shot_frame}\n")
            f.write(f"Shot Validity: {valid_shot_str}\n")
            
            if not is_valid_shot:
                f.write(f"Validation Failure Reason: {validation_data.get('failure_reason', 'unknown')}\n")
            else:
                f.write(f"Ball Max Height Frame: {validation_data['max_height_frame']}\n")
                f.write(f"Ball Height Above Rim: {-validation_data['max_height_diff']:.2f} pixels\n")
            
            f.write(f"Shooter ID: {shooter_id if shooter_id else 'Not detected'}\n")
            
            # Add enhanced shooter detection information
            if shooter_id is not None:
                f.write(f"Shooter Frame: {shooter_frame}\n")
                f.write(f"Shooter Detection Method: {shooter_detection_method}\n") 
                f.write(f"Shooter-Ball Distance: {shooter_ball_distance:.2f} pixels (threshold: {distance_threshold} pixels)\n")
            
            f.write(f"Jump Frame: {jump_frame if jump_frame else 'Not detected'}\n")
            f.write(f"Jump Detection Method: {detection_method}\n")
            f.write(f"Jersey Number: {jersey_number}\n")
            f.write(f"Team: {team_name}\n")
            f.write(f"Court Orientation: {court_orientation}\n")
            f.write(f"Team Match: {team_match_str}\n")
            f.write(f"Three-Point Shot: {three_point_str}\n")
            f.write(f"Distance to 3PT Line: {distance_str} pixels\n\n")
            
            if jump_frame and jump_position:
                f.write(f"Jump Position (y-coordinate): {jump_position}\n")
            
            if three_point_analysis:
                f.write("\nThree-Point Analysis Details:\n")
                for key, value in three_point_analysis.items():
                    f.write(f"  - {key}: {value}\n")
            
            f.write(f"\nAll analyzed frames: {all_frames}\n")
        
        # Create shot validation visualization for valid shots
        if is_valid_shot and validation_data['max_height_frame'] is not None:
            valid_shot_viz_path = shot_validation_dir / f'valid_shot_{shot_frame:05d}_max_height_{validation_data["max_height_frame"]}.png'
            create_visualization_with_court(court_h5_path, player_h5_path, validation_data['max_height_frame'], None, valid_shot_viz_path, 
                                          False, None, None, None, None, None, None, None, ball_data)
        
        # Determine target directory based on shot validity
        target_dir = sequences_dir if is_valid_shot else invalid_sequences_dir
        
        # For naming files - simplified as requested
        point_type = "3PT" if is_three_point else "2PT"
        
        if shooter_id is not None:
            if jump_frame is not None:
                print(f"Detected jump at frame {jump_frame}, foot position: {jump_position}")
                print(f"Detection method: {detection_method}")
                
                # Create sequence video with simplified naming
                sequence_output_path = target_dir / f'{jump_frame}_to_{shot_frame}_{team_type}_{point_type}.mp4'
                
                success, detected_team_type = create_jump_sequence_video(
                    court_h5_path, 
                    player_h5_path, 
                    jump_frame, 
                    shot_frame, 
                    shooter_id, 
                    sequence_output_path, 
                    ball_data,  # Pass ball data to the function
                    fps=10, 
                    pre_jump_frames=20, 
                    jersey_data=jersey_data, 
                    detection_method=detection_method,
                    shooter_frame=shooter_frame,
                    shooter_detection_method=shooter_detection_method,
                    shooter_ball_distance=shooter_ball_distance
                )
                
                if success:
                    print(f"Created shot sequence video: {sequence_output_path}")
                    print(f"Team type from sequence: {detected_team_type}")
                else:
                    print(f"Failed to create shot sequence video")
                
            else:
                print(f"Could not detect jump for shooter {shooter_id} (method: {detection_method})")
                
                # Even without a jump frame, create a video for both valid AND invalid shots
                # Use max lookback frames to create a sequence
                estimated_jump_frame = max(1, shot_frame - 30)  # Assume jump happened ~30 frames before shot
                
                # Create sequence video with estimated jump frame and simplified naming
                sequence_output_path = target_dir / f'{estimated_jump_frame}_to_{shot_frame}_{team_type}_{point_type}.mp4'
                
                success, detected_team_type = create_jump_sequence_video(
                    court_h5_path, 
                    player_h5_path, 
                    estimated_jump_frame, 
                    shot_frame, 
                    shooter_id, 
                    sequence_output_path, 
                    ball_data,  # Pass ball data to the function
                    fps=10, 
                    pre_jump_frames=10,  # Less pre-jump frames since we're already estimating
                    jersey_data=jersey_data, 
                    detection_method=detection_method,
                    shooter_frame=shooter_frame,
                    shooter_detection_method=shooter_detection_method,
                    shooter_ball_distance=shooter_ball_distance
                )
                
                if success:
                    print(f"Created shot sequence video (with estimated jump): {sequence_output_path}")
                    print(f"Team type from sequence: {detected_team_type}")
                else:
                    print(f"Failed to create shot sequence video")
        else:
            print(f"Could not find shooter for shot at frame {shot_frame}")
            
            # Even without finding a shooter, create a video for the shot (INVALID OR VALID)
            # Use preceding frames
            estimated_start_frame = max(1, shot_frame - 40)  # Look back 40 frames
            
            # Create a sequence video for shots with no detected shooter
            sequence_output_path = target_dir / f'{estimated_start_frame}_to_{shot_frame}_unknown_unknown.mp4'
            
            # Create a video with no specific shooter highlighted
            success, _ = create_jump_sequence_video(
                court_h5_path, 
                player_h5_path, 
                estimated_start_frame, 
                shot_frame, 
                None,  # No shooter to highlight
                sequence_output_path, 
                ball_data,  # Pass ball data to the function
                fps=10, 
                pre_jump_frames=10,
                jersey_data=jersey_data,
                detection_method="no_shooter_detected",
                shooter_frame=None,
                shooter_detection_method=None,
                shooter_ball_distance=None
            )
            
            if success:
                print(f"Created shot sequence video (no shooter): {sequence_output_path}")
            else:
                print(f"Failed to create shot sequence video")
            
            # Save the shot frame for reference
            output_path = target_dir / f'shot_no_shooter_{shot_frame:05d}_{validity_suffix}.png'
            create_visualization_with_court(court_h5_path, player_h5_path, shot_frame, None, output_path, 
                                          False, jersey_data, None, None, None, None, None, None, ball_data)
    
    # Print final scoring tally
    print("\n\n===== FINAL SCORING TALLY =====")
    print("\nTEAM SCORES:")
    print(f"HOME: {scoring_tally['home']['total']} points ({scoring_tally['home']['2PT']} × 2PT, {scoring_tally['home']['3PT']} × 3PT)")
    print(f"AWAY: {scoring_tally['away']['total']} points ({scoring_tally['away']['2PT']} × 2PT, {scoring_tally['away']['3PT']} × 3PT)")
    
    print("\nPLAYER SCORES:")
    for player_key, stats in scoring_tally["players"].items():
        team, identifier = player_key.split('_', 1)
        player_desc = f"#{identifier}" if identifier != "undetected" else "Undetected Player"
        print(f"{team.upper()} {player_desc}: {stats['total']} points ({stats['2PT']} × 2PT, {stats['3PT']} × 3PT)")
    
    # Write tally to file
    tally_file = Path(f'scoring_tally_{fileName}.txt')
    with open(tally_file, 'w') as f:
        f.write("===== FINAL SCORING TALLY =====\n\n")
        f.write("TEAM SCORES:\n")
        f.write(f"HOME: {scoring_tally['home']['total']} points ({scoring_tally['home']['2PT']} × 2PT, {scoring_tally['home']['3PT']} × 3PT)\n")
        f.write(f"AWAY: {scoring_tally['away']['total']} points ({scoring_tally['away']['2PT']} × 2PT, {scoring_tally['away']['3PT']} × 3PT)\n\n")
        
        f.write("PLAYER SCORES:\n")
        for player_key, stats in scoring_tally["players"].items():
            team, identifier = player_key.split('_', 1)
            player_desc = f"#{identifier}" if identifier != "undetected" else "Undetected Player"
            f.write(f"{team.upper()} {player_desc}: {stats['total']} points ({stats['2PT']} × 2PT, {stats['3PT']} × 3PT)\n")

if __name__ == "__main__":
    main()