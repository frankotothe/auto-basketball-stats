import cv2
import numpy as np
import h5py
from pathlib import Path
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def get_mask_from_player_h5(frame_group, object_key):
    """Extract mask from H5 player data using RLE decoding."""
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

def load_ball_detections(csv_path):
    """Load ball detection data from CSV file."""
    print(f"Loading ball detections from {csv_path}")
    start_time = time.time()
    
    ball_data = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                try:
                    frame_num = int(parts[0])
                    confidence = float(parts[3])
                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])
                    
                    # Only include ball detections with class_name 'ball'
                    if parts[2] == 'ball':
                        ball_data.append({
                            'frame_number': frame_num,
                            'confidence': confidence,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        })
                except (ValueError, IndexError) as e:
                    continue  # Skip problematic lines silently for speed
    
    # Sort by frame number
    ball_data.sort(key=lambda x: x['frame_number'])
    
    # Create frame-indexed dictionary
    raw_detections = {}
    for item in ball_data:
        frame_num = item['frame_number']
        if frame_num not in raw_detections:
            raw_detections[frame_num] = []
        raw_detections[frame_num].append(item)
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(ball_data)} ball detections in {elapsed_time:.2f} seconds")
    
    return raw_detections

def process_ball_tracking(raw_detections, max_frame):
    """Process ball tracking using the ball detection stabilization algorithm."""
    print(f"Processing ball tracking data for {max_frame} frames")
    start_time = time.time()
    
    # Parameters for tracking stability
    base_distance_threshold = 100
    high_conf_distance_threshold = 200
    high_conf_level = 0.7
    min_confidence = 0.4
    
    # Initialize stable ball state
    stable_ball = {
        'position': None,
        'radius': None,
        'confidence': 0,
        'stable_frames': 0,
        'last_seen_frame': 0,
        'lost_frames': 0,
        'potential_new': None
    }
    
    # Dictionary to store filtered ball positions
    filtered_detections = {}
    
    # Process each frame to create filtered ball positions
    for frame_num in range(1, max_frame + 1):
        # Get raw detections for this frame
        current_detections = raw_detections.get(frame_num, [])
        
        # Sort by confidence (highest first)
        current_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Update lost frames counter if no detections
        if not current_detections:
            stable_ball['lost_frames'] += 1
        else:
            # Process the highest confidence detection first
            best_detection = current_detections[0]
            current_x = best_detection['center_x']
            current_y = best_detection['center_y']
            current_conf = best_detection['confidence']
            current_radius = max(best_detection['width'], best_detection['height']) / 2
            
            # Determine distance threshold based on confidence
            current_distance_threshold = high_conf_distance_threshold if current_conf >= high_conf_level else base_distance_threshold
            
            # If we already have a stable position
            if stable_ball['position'] is not None:
                prev_x, prev_y = stable_ball['position']
                
                # Calculate distance from stable position
                distance = np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
                
                # If new detection is close enough to stable position
                if distance < current_distance_threshold:
                    # Update position with smoothing
                    weight_new = 0.5 if current_conf >= high_conf_level else 0.3
                        
                    new_x = prev_x * (1 - weight_new) + current_x * weight_new
                    new_y = prev_y * (1 - weight_new) + current_y * weight_new
                    
                    # Update stable ball state
                    stable_ball['position'] = (new_x, new_y)
                    stable_ball['radius'] = stable_ball['radius'] * (1 - weight_new) + current_radius * weight_new
                    stable_ball['confidence'] = current_conf
                    stable_ball['stable_frames'] += 1
                    stable_ball['last_seen_frame'] = frame_num
                    stable_ball['lost_frames'] = 0
                    stable_ball['potential_new'] = None
                    
                    # Add to filtered detections
                    filtered_detections[frame_num] = {
                        'center_x': new_x,
                        'center_y': new_y,
                        'radius': stable_ball['radius'],
                        'confidence': current_conf,
                        'status': 'stable'
                    }
                else:
                    # New detection is far from stable position
                    # High confidence detections are accepted immediately
                    if current_conf >= high_conf_level:
                        stable_ball['position'] = (current_x, current_y)
                        stable_ball['radius'] = current_radius
                        stable_ball['confidence'] = current_conf
                        stable_ball['stable_frames'] = 1
                        stable_ball['last_seen_frame'] = frame_num
                        stable_ball['lost_frames'] = 0
                        stable_ball['potential_new'] = None
                        
                        # Add to filtered detections
                        filtered_detections[frame_num] = {
                            'center_x': current_x,
                            'center_y': current_y,
                            'radius': current_radius,
                            'confidence': current_conf,
                            'status': 'high_conf_jump'
                        }
                    elif stable_ball['lost_frames'] >= 10:
                        # Ball missing for a while - lower threshold for accepting new positions
                        # Immediately accept medium confidence detection after long absence
                        if current_conf >= 0.5:
                            stable_ball['position'] = (current_x, current_y)
                            stable_ball['radius'] = current_radius
                            stable_ball['confidence'] = current_conf
                            stable_ball['stable_frames'] = 1
                            stable_ball['last_seen_frame'] = frame_num
                            stable_ball['lost_frames'] = 0
                            stable_ball['potential_new'] = None
                            
                            # Add to filtered detections
                            filtered_detections[frame_num] = {
                                'center_x': current_x,
                                'center_y': current_y,
                                'radius': current_radius,
                                'confidence': current_conf,
                                'status': 'reintroduced_after_absence'
                            }
                        else:
                            # Simplified potential new tracking for long absences
                            if stable_ball['potential_new'] is None:
                                stable_ball['potential_new'] = {
                                    'position': (current_x, current_y),
                                    'radius': current_radius,
                                    'confidence': current_conf,
                                    'count': 1,
                                    'first_seen': frame_num
                                }
                            else:
                                # Check if this detection is near the potential new position
                                pot_x, pot_y = stable_ball['potential_new']['position']
                                pot_dist = np.sqrt((current_x - pot_x)**2 + (current_y - pot_y)**2)
                                
                                if pot_dist < base_distance_threshold:
                                    # Update count and position
                                    stable_ball['potential_new']['count'] += 1
                                    
                                    # Only need 2 frames after long absence
                                    if stable_ball['potential_new']['count'] >= 2:
                                        # Accept as new position
                                        stable_ball['position'] = (current_x, current_y)
                                        stable_ball['radius'] = current_radius
                                        stable_ball['confidence'] = current_conf
                                        stable_ball['stable_frames'] = 1
                                        stable_ball['last_seen_frame'] = frame_num
                                        stable_ball['lost_frames'] = 0
                                        stable_ball['potential_new'] = None
                                        
                                        # Add to filtered detections
                                        filtered_detections[frame_num] = {
                                            'center_x': current_x,
                                            'center_y': current_y,
                                            'radius': current_radius,
                                            'confidence': current_conf,
                                            'status': 'reintroduced_confirmed'
                                        }
                                else:
                                    # Reset potential new
                                    stable_ball['potential_new'] = {
                                        'position': (current_x, current_y),
                                        'radius': current_radius,
                                        'confidence': current_conf,
                                        'count': 1,
                                        'first_seen': frame_num
                                    }
                                
                            # Increment lost frames counter
                            stable_ball['lost_frames'] += 1
                    else:
                        # Normal stability requirements
                        if stable_ball['potential_new'] is None:
                            stable_ball['potential_new'] = {
                                'position': (current_x, current_y),
                                'radius': current_radius,
                                'confidence': current_conf,
                                'count': 1,
                                'first_seen': frame_num
                            }
                        else:
                            # Check if this detection is close to potential new position
                            pot_x, pot_y = stable_ball['potential_new']['position']
                            pot_dist = np.sqrt((current_x - pot_x)**2 + (current_y - pot_y)**2)
                            
                            if pot_dist < base_distance_threshold:
                                # Update potential new position count
                                stable_ball['potential_new']['count'] += 1
                                
                                # Need 3 consecutive frames normally
                                if stable_ball['potential_new']['count'] >= 3:
                                    # Accept as new position
                                    stable_ball['position'] = (current_x, current_y)
                                    stable_ball['radius'] = current_radius
                                    stable_ball['confidence'] = current_conf
                                    stable_ball['stable_frames'] = 1
                                    stable_ball['last_seen_frame'] = frame_num
                                    stable_ball['lost_frames'] = 0
                                    stable_ball['potential_new'] = None
                                    
                                    # Add to filtered detections
                                    filtered_detections[frame_num] = {
                                        'center_x': current_x,
                                        'center_y': current_y,
                                        'radius': current_radius,
                                        'confidence': current_conf,
                                        'status': 'new_stable'
                                    }
                            else:
                                # Reset potential new if this is higher confidence
                                if current_conf > stable_ball['potential_new']['confidence']:
                                    stable_ball['potential_new'] = {
                                        'position': (current_x, current_y),
                                        'radius': current_radius,
                                        'confidence': current_conf,
                                        'count': 1,
                                        'first_seen': frame_num
                                    }
                        
                        # Increment lost frames counter
                        stable_ball['lost_frames'] += 1
            else:
                # No stable position yet - initialize with first detection
                if current_conf > min_confidence:
                    stable_ball['position'] = (current_x, current_y)
                    stable_ball['radius'] = current_radius
                    stable_ball['confidence'] = current_conf
                    stable_ball['stable_frames'] = 1
                    stable_ball['last_seen_frame'] = frame_num
                    stable_ball['lost_frames'] = 0
                    
                    # Add to filtered detections
                    filtered_detections[frame_num] = {
                        'center_x': current_x,
                        'center_y': current_y,
                        'radius': current_radius,
                        'confidence': current_conf,
                        'status': 'initial'
                    }
        
        # If we have a stable position but no detection this frame,
        # we can still show the last known position for a few frames
        if frame_num not in filtered_detections and stable_ball['position'] is not None:
            frames_since_seen = frame_num - stable_ball['last_seen_frame']
            
            # Only show previous position for a short time
            if frames_since_seen <= 3:
                # Add to filtered detections with decreased confidence
                filtered_detections[frame_num] = {
                    'center_x': stable_ball['position'][0],
                    'center_y': stable_ball['position'][1],
                    'radius': stable_ball['radius'],
                    'confidence': max(0.1, stable_ball['confidence'] - 0.15 * frames_since_seen),
                    'status': 'interpolated'
                }
    
    elapsed_time = time.time() - start_time
    print(f"Ball tracking processed in {elapsed_time:.2f} seconds")
    return filtered_detections

def preload_h5_data(h5_file_path, frame_start, frame_end):
    """Preload a batch of player data from H5 file for faster processing."""
    player_data_batch = {}
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            for frame_num in range(frame_start, frame_end + 1):
                h5_frame_key = f"frame_{frame_num:05d}"
                if h5_frame_key in h5_file:
                    player_data_batch[frame_num] = {}
                    frame_group = h5_file[h5_frame_key]
                    
                    # Get all player keys
                    player_keys = [k for k in frame_group.keys() if k.startswith('player')]
                    
                    for obj_key in player_keys:
                        # Store mask and attributes
                        player_data_batch[frame_num][obj_key] = {
                            'mask': get_mask_from_player_h5(frame_group, obj_key),
                            'attrs': dict(frame_group[obj_key].attrs)
                        }
    except Exception as e:
        print(f"Error preloading H5 data frames {frame_start}-{frame_end}: {e}")
    
    return player_data_batch

def process_video_frame_batch(frame_batch, player_data_batch, filtered_ball_detections,
                             video_path, width, height, player_color, ball_color,
                             use_video_background):
    """Process a batch of frames and return the rendered images."""
    rendered_frames = {}
    
    # Open video once for the entire batch if using video background
    video = None
    if use_video_background and video_path:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            use_video_background = False
    
    for frame_num in frame_batch:
        # Initialize frame
        if use_video_background and video:
            # Seek to the specific frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = video.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            elif frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
        else:
            # Create empty black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add player masks from preloaded data
        if frame_num in player_data_batch:
            frame_data = player_data_batch[frame_num]
            
            for obj_key, player_data in frame_data.items():
                # Get player mask
                player_mask = player_data['mask']
                if player_mask is None or not np.any(player_mask):
                    continue
                
                # Check if mask dimensions match the frame dimensions
                if player_mask.shape[0] != height or player_mask.shape[1] != width:
                    # Resize mask to match frame dimensions
                    player_mask_resized = cv2.resize(
                        player_mask.astype(np.uint8),
                        (width, height)
                    ).astype(bool)
                else:
                    player_mask_resized = player_mask
                
                # Get player attributes
                player_attrs = player_data['attrs']
                
                # Get centroid for text placement
                try:
                    centroid_x = int(float(player_attrs.get('centroid_x', 0)))
                    centroid_y = int(float(player_attrs.get('centroid_y', 0)))
                    
                    # Scale coordinates if mask was resized
                    if player_mask.shape[0] != height or player_mask.shape[1] != width:
                        scale_x = width / player_mask.shape[1]
                        scale_y = height / player_mask.shape[0]
                        centroid_x = int(centroid_x * scale_x)
                        centroid_y = int(centroid_y * scale_y)
                except:
                    # If centroid not available, calculate from mask
                    pixel_coords = np.where(player_mask_resized)
                    if len(pixel_coords[0]) > 0:
                        centroid_y = int(np.mean(pixel_coords[0]))
                        centroid_x = int(np.mean(pixel_coords[1]))
                    else:
                        continue
                
                # Apply player mask to frame (semi-transparent overlay)
                overlay = frame.copy()
                overlay[player_mask_resized] = player_color
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                # Add player ID text
                cv2.putText(frame, obj_key, (centroid_x, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add ball from filtered detections
        if frame_num in filtered_ball_detections:
            ball = filtered_ball_detections[frame_num]
            center_x = int(ball['center_x'])
            center_y = int(ball['center_y'])
            radius = max(5, int(ball['radius']))  # Ensure ball is visible
            status = ball['status']
            
            # Draw different circles based on status
            if status == 'interpolated':
                # Semi-transparent ball for interpolated positions
                overlay = frame.copy()
                cv2.circle(overlay, (center_x, center_y), radius, ball_color, -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            else:
                # Solid ball for detected positions
                cv2.circle(frame, (center_x, center_y), radius, ball_color, -1)
            
            # Add confidence text
            confidence_text = f"{ball['confidence']:.2f}"
            cv2.putText(frame, confidence_text, (center_x - radius, center_y - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        # Store rendered frame
        rendered_frames[frame_num] = frame
    
    # Release video capture if opened
    if video is not None:
        video.release()
    
    return rendered_frames

def create_player_ball_visualization_video(player_h5_path, ball_csv_path, output_path, video_path=None,
                                          width=1920, height=1080, fps=30):
    """
    Create a video visualization showing only player masks from H5 and ball from CSV tracking.
    Optimized version with proper resource management and batch processing.
    
    Args:
        player_h5_path: Path to the player tracking h5 file
        ball_csv_path: Path to the ball detection CSV file
        output_path: Path to save the output video
        video_path: Optional path to the original video for background
        width: Width of the output video
        height: Height of the output video
        fps: Frames per second of the output video
    """
    print(f"Starting video creation process...")
    total_start_time = time.time()
    
    # Determine maximum frame number from both H5 and video source
    max_frame = 0
    
    print("Scanning H5 file to determine frame range...")
    start_time = time.time()
    with h5py.File(player_h5_path, 'r') as h5_file:
        # Get all frame keys (format: frame_XXXXX)
        frame_keys = [k for k in h5_file.keys() if k.startswith('frame_')]
        if frame_keys:
            # Extract frame numbers and find the max
            frame_numbers = [int(k.split('_')[1]) for k in frame_keys]
            max_frame = max(frame_numbers) if frame_numbers else 0
    
    h5_scan_time = time.time() - start_time
    print(f"H5 file scanned in {h5_scan_time:.2f} seconds. Found {max_frame} frames.")
    
    # Check if we should use the original video as background
    use_video_background = video_path is not None
    if use_video_background:
        print("Checking video source...")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            use_video_background = False
        else:
            # Get video properties and ensure max_frame doesn't exceed video length
            video_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_total_frames > 0:
                max_frame = min(max_frame, video_total_frames) if max_frame > 0 else video_total_frames
            video.release()
    
    # Load and process ball detections
    raw_ball_detections = load_ball_detections(ball_csv_path)
    
    # Update max_frame to include ball detections as well
    if raw_ball_detections:
        ball_max_frame = max(raw_ball_detections.keys())
        max_frame = max(max_frame, ball_max_frame)
    
    print(f"Processing a total of {max_frame} frames")
    
    # Process ball tracking to get stabilized positions
    filtered_ball_detections = process_ball_tracking(raw_ball_detections, max_frame)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define colors for visualization (BGR format)
    player_color = (0, 255, 0)  # Green for players
    ball_color = (0, 165, 255)  # Orange for ball
    
    # Process in batches for better performance
    batch_size = 100  # Process 100 frames at a time
    
    # Determine number of CPU cores to use (leave one free for system)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for processing")
    
    for batch_start in range(1, max_frame + 1, batch_size):
        batch_start_time = time.time()
        batch_end = min(batch_start + batch_size - 1, max_frame)
        
        print(f"Processing batch: frames {batch_start} to {batch_end}")
        
        # Pre-load H5 data for this batch
        print("Preloading H5 data...")
        preload_start = time.time()
        player_data_batch = preload_h5_data(player_h5_path, batch_start, batch_end)
        preload_time = time.time() - preload_start
        print(f"Preloaded data for {len(player_data_batch)} frames in {preload_time:.2f} seconds")
        
        # Split batch into sub-batches for parallel processing
        frames_per_process = max(1, (batch_end - batch_start + 1) // num_cores)
        sub_batches = []
        
        for sub_batch_start in range(batch_start, batch_end + 1, frames_per_process):
            sub_batch_end = min(sub_batch_start + frames_per_process - 1, batch_end)
            sub_batches.append(list(range(sub_batch_start, sub_batch_end + 1)))
        
        # Process each sub-batch in parallel
        all_frames = {}
        
        if num_cores > 1:
            # Parallel processing
            print(f"Processing {len(sub_batches)} sub-batches in parallel...")
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                future_to_batch = {
                    executor.submit(
                        process_video_frame_batch, 
                        sub_batch, 
                        player_data_batch, 
                        filtered_ball_detections,
                        video_path,
                        width,
                        height,
                        player_color,
                        ball_color,
                        use_video_background
                    ): i for i, sub_batch in enumerate(sub_batches)
                }
                
                for future in as_completed(future_to_batch):
                    try:
                        batch_result = future.result()
                        all_frames.update(batch_result)
                    except Exception as e:
                        print(f"Error processing sub-batch {future_to_batch[future]}: {e}")
        else:
            # Serial processing
            print("Processing frames serially...")
            for sub_batch in sub_batches:
                batch_result = process_video_frame_batch(
                    sub_batch, 
                    player_data_batch, 
                    filtered_ball_detections,
                    video_path,
                    width,
                    height,
                    player_color,
                    ball_color,
                    use_video_background
                )
                all_frames.update(batch_result)
        
        # Write frames to video in order
        for frame_num in range(batch_start, batch_end + 1):
            if frame_num in all_frames:
                out.write(all_frames[frame_num])
        
        # Clear memory
        all_frames.clear()
        player_data_batch.clear()
        
        batch_time = time.time() - batch_start_time
        frames_per_second = (batch_end - batch_start + 1) / batch_time
        print(f"Batch completed in {batch_time:.2f} seconds ({frames_per_second:.2f} frames/second)")
        print(f"Batch memory usage released")
        
    # Release video writer
    out.release()
    
    total_time = time.time() - total_start_time
    print(f"Video processing complete! Total time: {total_time:.2f} seconds")
    print(f"Output saved to: {output_path}")

def main():
    # Input paths
    player_h5_path = "../trackingOutputs/2ndQIp.h5"
    ball_csv_path = "2ndQIpBall.csv"
    video_path = "../clips/2ndQIp.mp4"  # Original video for background (optional)
    
    # Output path
    output_path = "player_ball_visualization.mp4"
    
    # Create visualization video with original video as background
    create_player_ball_visualization_video(
        player_h5_path, 
        ball_csv_path, 
        output_path,
        #video_path  # Comment this line to use black background instead
    )
    
    print("Video processing complete!")

if __name__ == "__main__":
    main()