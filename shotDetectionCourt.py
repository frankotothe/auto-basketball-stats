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
    
    # First, identify the shooter by finding the player closest to the ball
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Find the shooter at shot frame
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
                closest_player_overall = closest_player
                object_id_overall = closest_object_id
                break
    
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
                bottom_pixels = sorted_y[:max(1, int(len(sorted_y) * 0.2))]
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

def create_visualization_with_court(court_h5_path, player_h5_path, frame_num, shooter_id, output_path, is_jump_frame=False, jersey_data=None, detection_method=None):
    default_height, default_width = 1080, 1920
    combined_canvas = np.zeros((default_height, default_width, 3), dtype=np.uint8)
    
    try:
        with h5py.File(court_h5_path, 'r') as court_h5:
            court_colors = {
                0: (200, 0, 0),   # Blue - Boundary
                1: (150, 0, 150),    # Green - Key
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
                    
                    # For jump frames, highlight the bottom 20% area of the shooter
                    if is_jump_frame and obj_key == shooter_id:
                        player_pixels = np.where(mask)
                        if len(player_pixels[0]) > 0:
                            # Sort y-positions in descending order (bottom to top)
                            y_positions = player_pixels[0]
                            sorted_indices = np.argsort(y_positions)[::-1]
                            
                            # Take bottom 20% of pixels
                            bottom_indices = sorted_indices[:max(1, int(len(sorted_indices) * 0.2))]
                            bottom_y = y_positions[bottom_indices]
                            bottom_x = player_pixels[1][bottom_indices]
                            
                            # Calculate average position
                            avg_y = int(np.mean(bottom_y))
                            x_at_avg_y = bottom_x[np.abs(bottom_y - avg_y).argmin()]
                            
                            # Draw a red rectangle around the bottom 20% area
                            min_x = np.min(bottom_x)
                            max_x = np.max(bottom_x)
                            min_y = np.min(bottom_y)
                            max_y = np.max(bottom_y)
                            
                            # Draw the bounding box of the area we're measuring
                            cv2.rectangle(combined_canvas, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                            
                            # Mark the average position with a red dot
                            cv2.circle(combined_canvas, (x_at_avg_y, avg_y), 5, (255, 0, 0), -1)
                            
                            # Add text indicating this is the jump point
                            avg_x = int(np.mean(bottom_x))
                            cv2.putText(combined_canvas, "JUMP AREA", (avg_x, max_y + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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
            create_visualization_with_court(
                court_h5_path, 
                player_h5_path, 
                frame_num, 
                current_shooter_id, 
                frame_path, 
                is_jump_frame, 
                jersey_data,
                detection_method if is_jump_frame else None
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

def main():
    fileName = "TestClip3"
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
    
    # Create a log file for detection methods
    log_file = detection_logs_dir / f'{fileName}_detection_methods.csv'
    with open(log_file, 'w') as f:
        f.write("shot_frame,jump_frame,shooter_id,detection_method,jersey_number\n")
    
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
    sequence_starts = find_sequence_starts(shots)
    
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
        
        # Log the detection information
        with open(log_file, 'a') as f:
            f.write(f"{shot_frame},{jump_frame if jump_frame else 'None'},{shooter_id if shooter_id else 'None'},{detection_method},{jersey_number}\n")
        
        # Create a detailed log for this shot
        shot_log_file = detection_logs_dir / f'shot_{shot_frame}_details.txt'
        with open(shot_log_file, 'w') as f:
            f.write(f"Shot Frame: {shot_frame}\n")
            f.write(f"Shooter ID: {shooter_id if shooter_id else 'Not detected'}\n")
            f.write(f"Jump Frame: {jump_frame if jump_frame else 'Not detected'}\n")
            f.write(f"Detection Method: {detection_method}\n")
            f.write(f"Jersey Number: {jersey_number}\n\n")
            
            if jump_frame and jump_position:
                f.write(f"Jump Position (y-coordinate): {jump_position}\n")
            
            f.write(f"\nAll analyzed frames: {all_frames}\n")
        
        if shooter_id is not None:
            if jump_frame is not None:
                print(f"Detected jump at frame {jump_frame}, foot position: {jump_position}")
                print(f"Detection method: {detection_method}")
                
                # Add detection method to the visualization filename
                jump_output_path = jump_detection_dir / f'jump_frame_{jump_frame:05d}_{detection_method}.png'
                create_visualization_with_court(court_h5_path, player_h5_path, jump_frame, shooter_id, jump_output_path, True, jersey_data, detection_method)
                
                # Create a video sequence from 20 frames before jump to shot
                sequence_output_path = sequences_dir / f'sequence_{jump_frame}_to_{shot_frame}_{detection_method}.mp4'
                success = create_jump_sequence_video(court_h5_path, player_h5_path, jump_frame, shot_frame, shooter_id, sequence_output_path, fps=10, pre_jump_frames=20, jersey_data=jersey_data, detection_method=detection_method)
                
                if success:
                    print(f"Created shot sequence video: {sequence_output_path}")
                else:
                    print(f"Failed to create shot sequence video")
                    
                # Also save the shot frame for reference
                shot_output_path = output_dir / f'made_shot_{shot_frame:05d}.png'
                create_visualization_with_court(court_h5_path, player_h5_path, shot_frame, shooter_id, shot_output_path, jersey_data=jersey_data)
                
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