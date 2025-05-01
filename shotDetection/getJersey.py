import numpy as np
import h5py
import cv2
from pathlib import Path
import os
import random
from sklearn.cluster import KMeans
from collections import defaultdict

def get_mask_from_player_h5(frame_group, object_key):
    """Extract mask from player H5 file"""
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

def extract_jersey_color(frame, mask, frame_shape):
    """
    Extract jersey color from a higher region of the player mask
    Handles mask and frame dimension mismatches
    Returns the average color and the jersey mask for visualization
    """
    # Resize mask if dimensions don't match
    if mask.shape[0] != frame_shape[0] or mask.shape[1] != frame_shape[1]:
        resized_mask = cv2.resize(mask.astype(np.uint8), (frame_shape[1], frame_shape[0])) > 0
    else:
        resized_mask = mask
    
    y_indices, x_indices = np.where(resized_mask)
    if len(y_indices) == 0:
        return None, None
        
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    height = max_y - min_y
    
    # Higher up on the player (from 5% to 35% from top)
    jersey_top = min_y + int(height * 0.20)
    jersey_bottom = min_y + int(height * 0.70)
    
    # Create jersey mask
    jersey_mask = np.zeros_like(resized_mask)
    jersey_mask[jersey_top:jersey_bottom, :] = resized_mask[jersey_top:jersey_bottom, :]
    
    # Extract colors from the jersey region
    jersey_pixels = frame[jersey_mask]
    
    if len(jersey_pixels) < 10:  # Need enough pixels
        return None, None
        
    # Calculate average color
    avg_color = np.mean(jersey_pixels, axis=0)
    
    # Get bounding box of the jersey region for cropping
    y_jersey_indices, x_jersey_indices = np.where(jersey_mask)
    if len(y_jersey_indices) == 0:
        return avg_color, None
        
    min_y_jersey = np.min(y_jersey_indices)
    max_y_jersey = np.max(y_jersey_indices)
    min_x_jersey = np.min(x_jersey_indices)
    max_x_jersey = np.max(x_jersey_indices)
    
    # Crop to the jersey region
    jersey_crop = frame[min_y_jersey:max_y_jersey, min_x_jersey:max_x_jersey].copy()
    
    # Add outline to show where jersey region is
    jersey_mask_crop = jersey_mask[min_y_jersey:max_y_jersey, min_x_jersey:max_x_jersey]
    jersey_outline = jersey_crop.copy()
    jersey_outline[np.logical_not(jersey_mask_crop)] = [0, 0, 0]  # Make non-jersey pixels black
    
    # Create a small red border around the jersey pixels
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(jersey_mask_crop.astype(np.uint8), kernel, iterations=1)
    border = dilated - jersey_mask_crop.astype(np.uint8)
    jersey_outline[border.astype(bool)] = [0, 0, 255]  # Red border
    
    return avg_color, jersey_outline

def classify_jersey_colors(frame, player_masks):
    """Classify jersey colors into light and dark teams using K-means clustering with confidence scores"""
    colors = []
    player_ids = []
    jersey_regions = {}
    frame_shape = frame.shape[:2]  # (height, width)
    
    # Extract average colors for each player
    for player_id, mask in player_masks.items():
        avg_color, jersey_region = extract_jersey_color(frame, mask, frame_shape)
        if avg_color is not None:
            colors.append(avg_color)
            player_ids.append(player_id)
            jersey_regions[player_id] = jersey_region
    
    if len(colors) < 2:
        return {}, None, None, {}  # Not enough players to classify
    
    # Convert to HSV for better color analysis
    colors_array = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
    hsv_colors = cv2.cvtColor(colors_array, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Use K-means to cluster into 2 teams
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(hsv_colors)
    
    # Determine which cluster is "light" and which is "dark" based on V value in HSV
    cluster_centers = kmeans.cluster_centers_
    cluster0_brightness = cluster_centers[0][2]  # V value in HSV
    cluster1_brightness = cluster_centers[1][2]
    
    if cluster0_brightness > cluster1_brightness:
        light_cluster, dark_cluster = 0, 1
    else:
        light_cluster, dark_cluster = 1, 0
    
    # Calculate distances to both cluster centers for confidence scoring
    distances_to_centers = np.zeros((len(hsv_colors), 2))
    for i in range(len(hsv_colors)):
        distances_to_centers[i, 0] = np.linalg.norm(hsv_colors[i] - cluster_centers[0])
        distances_to_centers[i, 1] = np.linalg.norm(hsv_colors[i] - cluster_centers[1])
    
    # Assign team labels to players with confidence scores
    team_assignments = {}
    for i, player_id in enumerate(player_ids):
        # Calculate confidence based on relative distance to assigned cluster vs other cluster
        dist_to_assigned = distances_to_centers[i, cluster_labels[i]]
        dist_to_other = distances_to_centers[i, 1 - cluster_labels[i]]
        
        # Confidence formula: how much closer to assigned cluster vs total distance to both
        # This gives a value between 0.5 (equidistant) and 1.0 (very confident)
        confidence = dist_to_other / (dist_to_assigned + dist_to_other)
        
        if cluster_labels[i] == light_cluster:
            team_assignments[player_id] = {"team": "Light Team", "confidence": confidence}
        else:
            team_assignments[player_id] = {"team": "Dark Team", "confidence": confidence}
            
    return team_assignments, (light_cluster, dark_cluster), cluster_centers, jersey_regions

def get_player_centroids(player_masks, frame_shape):
    """Get centroids for each player, with mask resizing if needed"""
    centroids = {}
    
    for player_id, mask in player_masks.items():
        # Resize mask if dimensions don't match
        if mask.shape[0] != frame_shape[0] or mask.shape[1] != frame_shape[1]:
            resized_mask = cv2.resize(mask.astype(np.uint8), (frame_shape[1], frame_shape[0])) > 0
        else:
            resized_mask = mask
        
        y_indices, x_indices = np.where(resized_mask)
        if len(y_indices) == 0:
            continue
            
        # Calculate centroid
        cy = int(np.mean(y_indices))
        cx = int(np.mean(x_indices))
        centroids[player_id] = (cx, cy)
    
    return centroids

def get_random_frame_with_players(player_h5_path, min_players=6, min_frame_buffer=2):
    """
    Get a random frame from the H5 file that has at least min_players
    and has at least min_frame_buffer frames before and after
    """
    valid_frames = []
    
    with h5py.File(player_h5_path, 'r') as h5_file:
        # Get all frame keys
        frame_keys = [key for key in h5_file.keys() if key.startswith('frame_')]
        frame_numbers = sorted([int(key.split('_')[1]) for key in frame_keys])
        
        # Find frames with enough players
        for frame_key in frame_keys:
            frame_group = h5_file[frame_key]
            player_count = sum(1 for key in frame_group.keys() if key.startswith('player_'))
            
            frame_num = int(frame_key.split('_')[1])
            
            # Check if there are enough frames before and after
            frame_idx = frame_numbers.index(frame_num)
            if (frame_idx >= min_frame_buffer and 
                frame_idx < len(frame_numbers) - min_frame_buffer and
                player_count >= min_players):
                valid_frames.append(frame_num)
    
    if not valid_frames:
        return None
        
    # Pick a random frame from valid ones
    return random.choice(valid_frames)

def get_player_masks_for_frame(h5_file, frame_num):
    """Get all player masks for a specific frame"""
    player_masks = {}
    frame_key = f"frame_{frame_num:05d}"
    
    if frame_key not in h5_file:
        return player_masks
        
    frame_group = h5_file[frame_key]
    
    for key in frame_group.keys():
        if key.startswith('player_'):
            mask = get_mask_from_player_h5(frame_group, key)
            if mask is not None and np.any(mask):
                player_masks[key] = mask
                
    return player_masks

def get_temporal_team_assignments(player_h5_path, video_path, frame_num, window_size=2):
    """
    Analyze player team assignments across multiple frames 
    (frame_num and window_size frames before and after)
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return {}, {}
    
    frame_shape = None
    
    # Get frames to analyze
    frames_to_analyze = list(range(frame_num - window_size, frame_num + window_size + 1))
    
    # Store all player colors across frames
    player_colors_across_frames = defaultdict(list)
    player_team_votes = defaultdict(lambda: {"Light Team": 0, "Dark Team": 0})
    player_confidence_sum = defaultdict(float)
    player_frames_count = defaultdict(int)
    
    # Store jersey regions for visualization (from main frame only)
    central_jersey_regions = {}
    
    # First classification to get cluster centers
    with h5py.File(player_h5_path, 'r') as h5_file:
        # Get main frame for initial classification
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, central_frame = video.read()
        if not success:
            video.release()
            return {}, {}
            
        frame_shape = central_frame.shape[:2]
        
        # Get player masks for the central frame
        central_player_masks = get_player_masks_for_frame(h5_file, frame_num)
        
        # Initial classification to get cluster centers
        initial_assignments, cluster_info, cluster_centers, jersey_regions = classify_jersey_colors(
            central_frame, central_player_masks)
        
        # Save jersey regions for the main frame
        central_jersey_regions = jersey_regions
        
        light_cluster, dark_cluster = cluster_info
        
        # Now analyze all frames in the window
        for f_num in frames_to_analyze:
            video.set(cv2.CAP_PROP_POS_FRAMES, f_num)
            success, frame = video.read()
            
            if not success:
                continue
                
            # Get player masks for this frame
            player_masks = get_player_masks_for_frame(h5_file, f_num)
            
            # For each player in this frame
            for player_id, mask in player_masks.items():
                avg_color, _ = extract_jersey_color(frame, mask, frame_shape)
                if avg_color is not None:
                    # Convert to HSV
                    color_array = np.array([avg_color], dtype=np.uint8).reshape(-1, 1, 3)
                    hsv_color = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV).reshape(-1, 3)[0]
                    
                    # Calculate distances to both cluster centers
                    dist_to_light = np.linalg.norm(hsv_color - cluster_centers[light_cluster])
                    dist_to_dark = np.linalg.norm(hsv_color - cluster_centers[dark_cluster])
                    
                    # Determine team and confidence
                    if dist_to_light < dist_to_dark:
                        team = "Light Team"
                        confidence = dist_to_dark / (dist_to_light + dist_to_dark)
                    else:
                        team = "Dark Team"
                        confidence = dist_to_light / (dist_to_light + dist_to_dark)
                    
                    # Record vote and confidence
                    player_team_votes[player_id][team] += 1
                    player_confidence_sum[player_id] += confidence
                    player_frames_count[player_id] += 1
    
    video.release()
    
    # Determine final team assignments with temporal consistency
    temporal_team_assignments = {}
    
    for player_id, votes in player_team_votes.items():
        if player_frames_count[player_id] == 0:
            continue
            
        # Determine team by majority vote
        if votes["Light Team"] > votes["Dark Team"]:
            team = "Light Team"
            vote_confidence = votes["Light Team"] / player_frames_count[player_id]
        else:
            team = "Dark Team"
            vote_confidence = votes["Dark Team"] / player_frames_count[player_id]
        
        # Average confidence across frames
        avg_confidence = player_confidence_sum[player_id] / player_frames_count[player_id]
        
        # Combined confidence: average of vote confidence and color confidence
        combined_confidence = (vote_confidence + avg_confidence) / 2
        
        temporal_team_assignments[player_id] = {
            "team": team, 
            "confidence": combined_confidence,
            "frames_tracked": player_frames_count[player_id],
            "total_frames": len(frames_to_analyze)
        }
    
    return temporal_team_assignments, central_jersey_regions

def create_original_visualization(frame, team_assignments, player_centroids, frame_num):
    """Create the original visualization with just team labels"""
    vis_frame = frame.copy()
    
    # Draw team labels for each player
    for player_id, team_info in team_assignments.items():
        if player_id not in player_centroids:
            continue
            
        team = team_info["team"]
        confidence = team_info["confidence"]
        frames_tracked = team_info["frames_tracked"]
        total_frames = team_info["total_frames"]
        cx, cy = player_centroids[player_id]
        
        # Get color for label text (white for dark team, black for light team)
        text_color = (0, 0, 0) if team == "Light Team" else (255, 255, 255)
        bg_color = (255, 255, 255) if team == "Light Team" else (0, 0, 0)
        
        # Draw text background
        text = f"{team} ({confidence:.2f})"
        track_text = f"Tracked: {frames_tracked}/{total_frames}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        track_text_size = cv2.getTextSize(track_text, font, 0.4, 1)[0]
        
        # Draw rectangle background for team
        cv2.rectangle(vis_frame, 
                     (cx - text_size[0]//2 - 5, cy - text_size[1] - 5),
                     (cx + text_size[0]//2 + 5, cy - 5),
                     bg_color, -1)
        
        # Draw rectangle background for tracking info
        cv2.rectangle(vis_frame, 
                     (cx - track_text_size[0]//2 - 5, cy + 5),
                     (cx + track_text_size[0]//2 + 5, cy + track_text_size[1] + 15),
                     (50, 50, 50), -1)
        
        # Draw text
        cv2.putText(vis_frame, text, (cx - text_size[0]//2, cy - 10), 
                    font, 0.5, text_color, 2)
        
        # Draw tracking info
        cv2.putText(vis_frame, track_text, (cx - track_text_size[0]//2, cy + 20), 
                    font, 0.4, (255, 255, 255), 1)
        
        # Also draw player ID above the player
        cv2.putText(vis_frame, player_id, (cx, cy - 30), 
                    font, 0.5, (0, 255, 255), 2)
    
    # Add frame information
    cv2.putText(vis_frame, f"Frame: {frame_num} (with ±2 frame analysis)", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Show team statistics
    light_count = sum(1 for team_info in team_assignments.values() if team_info["team"] == "Light Team")
    dark_count = sum(1 for team_info in team_assignments.values() if team_info["team"] == "Dark Team")
    
    cv2.putText(vis_frame, f"Light Team: {light_count} players", (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Dark Team: {dark_count} players", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    
    return vis_frame

def create_jersey_region_visualization(frame, team_assignments, player_centroids, frame_num, jersey_regions):
    """Create visualization with jersey regions and average team colors displayed at the top"""
    vis_frame = frame.copy()
    
    # Calculate average RGB color for each team (for visualization)
    light_team_colors = []
    dark_team_colors = []
    
    for player_id, team_info in team_assignments.items():
        if player_id in jersey_regions and jersey_regions[player_id] is not None:
            if team_info["team"] == "Light Team":
                light_team_colors.append(np.mean(jersey_regions[player_id], axis=(0, 1)))
            else:
                dark_team_colors.append(np.mean(jersey_regions[player_id], axis=(0, 1)))
    
    # Calculate average colors
    light_team_avg_color = np.mean(light_team_colors, axis=0) if light_team_colors else np.array([200, 200, 200])
    dark_team_avg_color = np.mean(dark_team_colors, axis=0) if dark_team_colors else np.array([50, 50, 50])
    
    # Convert average colors to integers for display
    light_color_display = tuple(int(c) for c in light_team_avg_color)
    dark_color_display = tuple(int(c) for c in dark_team_avg_color)
    
    # Display team colors at the top of the frame
    color_box_size = 60
    padding = 10
    
    # Background for color display area
    cv2.rectangle(vis_frame, (0, 0), (frame.shape[1], color_box_size*2 + padding*3), (0, 0, 0), -1)
    
    # Draw color boxes and labels
    # Light team
    cv2.rectangle(vis_frame, 
                 (padding, padding), 
                 (padding + color_box_size, padding + color_box_size), 
                 light_color_display, -1)
    cv2.putText(vis_frame, 
                f"Light Team RGB: {light_color_display}", 
                (padding*2 + color_box_size, padding + color_box_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Dark team
    cv2.rectangle(vis_frame, 
                 (padding, padding*2 + color_box_size), 
                 (padding + color_box_size, padding*2 + color_box_size*2), 
                 dark_color_display, -1)
    cv2.putText(vis_frame, 
                f"Dark Team RGB: {dark_color_display}", 
                (padding*2 + color_box_size, padding*2 + color_box_size + color_box_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw HSV values as well
    light_hsv = cv2.cvtColor(np.uint8([[light_color_display]]), cv2.COLOR_BGR2HSV)[0][0]
    dark_hsv = cv2.cvtColor(np.uint8([[dark_color_display]]), cv2.COLOR_BGR2HSV)[0][0]
    
    cv2.putText(vis_frame, 
                f"HSV: ({light_hsv[0]}, {light_hsv[1]}, {light_hsv[2]})", 
                (padding*2 + color_box_size + 300, padding + color_box_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(vis_frame, 
                f"HSV: ({dark_hsv[0]}, {dark_hsv[1]}, {dark_hsv[2]})", 
                (padding*2 + color_box_size + 300, padding*2 + color_box_size + color_box_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw team labels for each player
    for player_id, team_info in team_assignments.items():
        if player_id not in player_centroids:
            continue
            
        team = team_info["team"]
        confidence = team_info["confidence"]
        frames_tracked = team_info["frames_tracked"]
        total_frames = team_info["total_frames"]
        cx, cy = player_centroids[player_id]
        
        # Get color for label text (white for dark team, black for light team)
        text_color = (0, 0, 0) if team == "Light Team" else (255, 255, 255)
        bg_color = (255, 255, 255) if team == "Light Team" else (0, 0, 0)
        
        # Draw text background
        text = f"{team} ({confidence:.2f})"
        track_text = f"Tracked: {frames_tracked}/{total_frames}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        track_text_size = cv2.getTextSize(track_text, font, 0.4, 1)[0]
        
        # Draw rectangle background for team
        cv2.rectangle(vis_frame, 
                     (cx - text_size[0]//2 - 5, cy - text_size[1] - 5),
                     (cx + text_size[0]//2 + 5, cy - 5),
                     bg_color, -1)
        
        # Draw rectangle background for tracking info
        cv2.rectangle(vis_frame, 
                     (cx - track_text_size[0]//2 - 5, cy + 5),
                     (cx + track_text_size[0]//2 + 5, cy + track_text_size[1] + 15),
                     (50, 50, 50), -1)
        
        # Draw text
        cv2.putText(vis_frame, text, (cx - text_size[0]//2, cy - 10), 
                    font, 0.5, text_color, 2)
        
        # Draw tracking info
        cv2.putText(vis_frame, track_text, (cx - track_text_size[0]//2, cy + 20), 
                    font, 0.4, (255, 255, 255), 1)
        
        # Also draw player ID above the player
        cv2.putText(vis_frame, player_id, (cx, cy - 30), 
                    font, 0.5, (0, 255, 255), 2)
        
        # Draw jersey region visualization next to player
        if player_id in jersey_regions and jersey_regions[player_id] is not None:
            jersey_img = jersey_regions[player_id]
            jersey_height, jersey_width = jersey_img.shape[:2]
            
            # Scale jersey region if it's too small
            scale_factor = max(1, 40 / max(jersey_height, jersey_width))
            display_width = int(jersey_width * scale_factor)
            display_height = int(jersey_height * scale_factor)
            
            jersey_display = cv2.resize(jersey_img, (display_width, display_height))
            
            # Calculate position for jersey region (to the right of player ID)
            jersey_x = cx + 20
            jersey_y = cy - 60
            
            # Ensure jersey region fits within frame boundaries
            max_x = min(jersey_x + display_width, vis_frame.shape[1] - 1)
            max_y = min(jersey_y + display_height, vis_frame.shape[0] - 1)
            
            # Adjust dimensions if needed
            display_width = max_x - jersey_x
            display_height = max_y - jersey_y
            
            if display_width > 0 and display_height > 0:
                # Draw a border around the jersey region
                cv2.rectangle(vis_frame, 
                             (jersey_x - 2, jersey_y - 2),
                             (jersey_x + display_width + 2, jersey_y + display_height + 2),
                             (0, 255, 255), 1)
                
                # Add the jersey region to the visualization
                vis_frame[jersey_y:jersey_y+display_height, jersey_x:jersey_x+display_width] = \
                    jersey_display[:display_height, :display_width]
    
    # Add frame information
    cv2.putText(vis_frame, f"Frame: {frame_num} (with ±2 frame analysis)", (30, color_box_size*2 + padding*4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Show team statistics
    light_count = sum(1 for team_info in team_assignments.values() if team_info["team"] == "Light Team")
    dark_count = sum(1 for team_info in team_assignments.values() if team_info["team"] == "Dark Team")
    
    # Draw team info with actual jersey colors
    stats_y = color_box_size*2 + padding*6
    cv2.rectangle(vis_frame, (30, stats_y), (200, stats_y + 20), light_color_display, -1)
    cv2.putText(vis_frame, f"Light Team: {light_count} players", (35, stats_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.rectangle(vis_frame, (30, stats_y + 30), (200, stats_y + 50), dark_color_display, -1)
    cv2.putText(vis_frame, f"Dark Team: {dark_count} players", (35, stats_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Explanation of jersey region
    cv2.putText(vis_frame, "Jersey regions used for classification shown next to each player", (30, stats_y + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis_frame

def main():
    """Main function to run the jersey color classifier with temporal consistency"""
    # File paths - adjust as needed
    fileName = "2ndQIp"
    player_h5_path = f'../trackingOutputs/{fileName}.h5'
    video_path = f'../clips/{fileName}.mp4'
    output_dir = Path('team_classification')
    output_dir.mkdir(exist_ok=True)
    
    # Get a random frame with enough players and buffer frames
    frame_num = get_random_frame_with_players(player_h5_path, min_frame_buffer=2)
    if frame_num is None:
        print("Could not find a suitable frame with enough players and buffer frames")
        return
    
    print(f"Selected frame {frame_num} for team classification")
    
    # Get the actual video frame to know its dimensions
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return
        
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = video.read()
    video.release()
    
    if not success:
        print(f"Could not read frame {frame_num} from video")
        return
    
    frame_shape = frame.shape[:2]  # (height, width)
    print(f"Video frame dimensions: {frame_shape}")
    
    # Get temporal team assignments and jersey regions
    team_assignments, jersey_regions = get_temporal_team_assignments(player_h5_path, video_path, frame_num, window_size=2)
    
    if not team_assignments:
        print("Could not classify teams - not enough valid jersey colors")
        return
    
    # Get player masks for the main frame to visualize
    player_masks = {}
    with h5py.File(player_h5_path, 'r') as h5_file:
        frame_key = f"frame_{frame_num:05d}"
        if frame_key not in h5_file:
            print(f"Frame {frame_num} not found in H5 file")
            return
            
        frame_group = h5_file[frame_key]
        
        for key in frame_group.keys():
            if key.startswith('player_'):
                mask = get_mask_from_player_h5(frame_group, key)
                if mask is not None and np.any(mask):
                    player_masks[key] = mask
    
    # Get player centroids with proper resizing
    player_centroids = get_player_centroids(player_masks, frame_shape)
    
    # Create both types of visualizations
    original_vis = create_original_visualization(frame, team_assignments, player_centroids, frame_num)
    jersey_region_vis = create_jersey_region_visualization(frame, team_assignments, player_centroids, frame_num, jersey_regions)
    
    # Save both visualizations
    original_output_path = output_dir / f'temporal_team_classification_frame_{frame_num}.png'
    jersey_region_output_path = output_dir / f'jersey_region_team_classification_frame_{frame_num}.png'
    
    cv2.imwrite(str(original_output_path), original_vis)
    cv2.imwrite(str(jersey_region_output_path), jersey_region_vis)
    
    print(f"Original visualization saved to: {original_output_path}")
    print(f"Jersey region visualization saved to: {jersey_region_output_path}")
    
    # Optional: Save the raw frame for comparison
    raw_output_path = output_dir / f'raw_frame_{frame_num}.png'
    cv2.imwrite(str(raw_output_path), frame)
    
    # Print team assignments for all players
    print("\nTemporal Team Classifications with Confidence:")
    for player_id, team_info in sorted(team_assignments.items()):
        print(f"{player_id}: {team_info['team']} (Confidence: {team_info['confidence']:.2f}, " +
              f"Tracked: {team_info['frames_tracked']}/{team_info['total_frames']} frames)")

if __name__ == "__main__":
    main()