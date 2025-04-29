import cv2
import numpy as np

def create_ball_visualization_video(ball_csv_path, video_path, output_path, width=None, height=None, fps=30):
    # Load raw data
    ball_data = []
    with open(ball_csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                try:
                    frame_num = int(parts[0])
                    timestamp = float(parts[1])
                    class_name = parts[2]
                    confidence = float(parts[3])
                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])
                    
                    # Only include ball detections
                    if class_name == 'ball':
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
                    print(f"Skipping line due to error: {e}")
    
    # Sort by frame number
    ball_data.sort(key=lambda x: x['frame_number'])
    
    # Create frame-indexed dictionary
    raw_detections = {}
    for item in ball_data:
        frame_num = item['frame_number']
        if frame_num not in raw_detections:
            raw_detections[frame_num] = []
        raw_detections[frame_num].append(item)
    
    # Open the video to get dimensions and total frames
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    if width is None or height is None:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use video fps if not specified
    if fps is None:
        fps = video_fps
    
    # Find max frame number for ball detections
    max_frame = max(raw_detections.keys()) if raw_detections else total_frames
    max_frame = min(max_frame, total_frames)  # Don't exceed video frame count
    
    # Parameters for tracking stability
    base_distance_threshold = 100      # Base distance threshold (for normal confidence)
    high_conf_distance_threshold = 200  # Higher threshold for high confidence detections
    high_conf_level = 0.7              # Confidence level considered "high"
    min_confidence = 0.4               # Minimum confidence to consider for new positions
    min_stable_frames = 3              # Minimum consecutive frames to confirm a new position
    lost_track_threshold = 10          # Frames without detection to consider tracking lost
    
    # Initialize stable ball state
    stable_ball = {
        'position': None,      # (x, y)
        'radius': None,        # Ball radius 
        'confidence': 0,       # Current confidence
        'stable_frames': 0,    # How many frames it's been stable
        'last_seen_frame': 0,  # Last frame where it was detected
        'lost_frames': 0,      # How many frames without detection
        'potential_new': None  # Potential new position being evaluated
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
            # Higher confidence = allow bigger movements
            current_distance_threshold = high_conf_distance_threshold if current_conf >= high_conf_level else base_distance_threshold
            
            # If we already have a stable position
            if stable_ball['position'] is not None:
                prev_x, prev_y = stable_ball['position']
                
                # Calculate distance from stable position
                distance = np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
                
                # If new detection is close enough to stable position (using dynamic threshold)
                if distance < current_distance_threshold:
                    # Update position with smoothing (70% previous, 30% new)
                    # For high confidence detections, use more weight on the new position
                    if current_conf >= high_conf_level:
                        weight_new = 0.5  # More weight on new position for high confidence
                    else:
                        weight_new = 0.3  # Standard weight
                        
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
                        'radius': stable_ball['radius'] * 0.2,  # Make ball 80% smaller
                        'confidence': current_conf,
                        'status': 'stable'
                    }
                else:
                    # New detection is far from stable position
                    # For high confidence detections, accept more readily
                    if current_conf >= high_conf_level:
                        # Accept high confidence detection immediately
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
                            'radius': current_radius * 0.2,  # Make ball 80% smaller
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
                                'radius': current_radius * 0.2,  # Make ball 80% smaller
                                'confidence': current_conf,
                                'status': 'reintroduced_after_absence'
                            }
                        else:
                            # Start potential new with less strict requirements
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
                                            'radius': current_radius * 0.2,  # Make ball 80% smaller
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
                                        'radius': current_radius * 0.2,  # Make ball 80% smaller
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
                        'radius': current_radius * 0.2,  # Make ball 80% smaller
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
                    'radius': stable_ball['radius'] * 0.2,  # Make ball 80% smaller
                    'confidence': max(0.1, stable_ball['confidence'] - 0.15 * frames_since_seen),
                    'status': 'interpolated'
                }
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Orange color in BGR
    orange_color = (0, 165, 255)
    
    # Reset video to first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process each frame
    for frame_num in range(1, max_frame + 1):
        # Read frame from video
        ret, frame = video.read()
        if not ret:
            print(f"Error reading frame {frame_num} from video")
            break
        
        # Resize frame if necessary
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_num}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        # Draw raw detections as small circles
        if frame_num in raw_detections:
            for detection in raw_detections[frame_num]:
                center_x = int(detection['center_x'])
                center_y = int(detection['center_y'])
                radius = int(max(detection['width'], detection['height']) / 2) * 0.2  # 80% smaller
                
                # Draw small circle for raw detections
                cv2.circle(frame, (center_x, center_y), max(2, int(radius)), (0, 0, 255), 1)
                
                # Add confidence text
                confidence_text = f"{detection['confidence']:.2f}"
                cv2.putText(frame, confidence_text, (center_x + 10, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw filtered ball
        if frame_num in filtered_detections:
            ball = filtered_detections[frame_num]
            center_x = int(ball['center_x'])
            center_y = int(ball['center_y'])
            radius = int(ball['radius'])
            status = ball['status']
            
            # Draw different circles based on status
            if status == 'interpolated':
                # Semi-transparent orange circle for interpolated positions
                overlay = frame.copy()
                cv2.circle(overlay, (center_x, center_y), radius, orange_color, -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            elif status == 'initial':
                # Hollow orange circle for initial detections
                cv2.circle(frame, (center_x, center_y), radius, orange_color, 2)
            else:
                # Solid orange circle for all other stable positions
                cv2.circle(frame, (center_x, center_y), radius, orange_color, -1)
            
            # Add confidence text
            confidence_text = f"{ball['confidence']:.2f}"
            cv2.putText(frame, confidence_text, (center_x - radius, center_y - radius - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add status text
            cv2.putText(frame, status, (center_x - radius, center_y - radius - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Write frame to output video
        out.write(frame)
        
        # Print progress
        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{max_frame} frames")
    
    # Release resources
    video.release()
    out.release()
    print(f"Video saved to {output_path}")

# Use your specific CSV file and video
if __name__ == "__main__":
    ball_csv_path = "2ndQIpBall.csv"
    video_path = "../clips/2ndQIp.mp4"  # Path to the original video
    output_path = "ball_tracking_overlay.mp4"
    
    create_ball_visualization_video(ball_csv_path, video_path, output_path)