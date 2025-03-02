import h5py
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import os
import multiprocessing as mp
from tqdm import tqdm
import time

def rle_decode(rle, shape):
    """Decode run-length encoded mask"""
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    for start, end in rle:
        mask[start:end] = True
    return mask.reshape(shape)

def get_frame_info(h5_file):
    """Extract frame information from H5 file efficiently"""
    with h5py.File(h5_file, 'r') as f:
        if 'frames' not in f:
            return {'shape': None, 'valid_frames': []}
        
        # Get all frame numbers
        frame_keys = sorted([k for k in f['frames'].keys() if k.startswith('frame_')])
        if not frame_keys:
            return {'shape': None, 'valid_frames': []}
        
        # Get frame shape from first valid frame
        for key in frame_keys:
            frame = f['frames'][key]
            if not list(frame.keys()):
                continue
                
            detection = frame[list(frame.keys())[0]]
            if 'rle' in detection and 'shape' in detection['rle'].attrs:
                shape = tuple(detection['rle'].attrs['shape'])
                break
        else:
            return {'shape': None, 'valid_frames': []}
            
        # Extract valid frame numbers
        valid_frames = [int(k.split('_')[1]) for k in frame_keys]
        
        return {'shape': shape, 'valid_frames': valid_frames}

def load_frame_data(h5_file, frame_number):
    """Load data for a single frame without reopening the file"""
    with h5py.File(h5_file, 'r') as f:
        frame_key = f'frame_{frame_number}'
        if frame_key not in f['frames']:
            return None
            
        frame = f['frames'][frame_key]
        if not list(frame.keys()):
            return None
            
        frame_data = {'masks': {}}
        
        # Load shape information once
        first_detection = frame[list(frame.keys())[0]]
        mask_shape = tuple(first_detection['rle'].attrs['shape'])
        frame_data['shape'] = mask_shape
        
        # Process all detections in the frame
        for detection_key in frame.keys():
            detection = frame[detection_key]
            class_id = detection.attrs['class_id']
            
            # Decode RLE data for the mask
            mask = rle_decode(detection['rle'][:], mask_shape)
            
            # Store by class ID
            if class_id not in frame_data['masks']:
                frame_data['masks'][class_id] = np.zeros(mask_shape, dtype=bool)
            frame_data['masks'][class_id] |= mask
            
        return frame_data

def sort_points_vectorized(points):
    """Faster vectorized implementation of point sorting"""
    if len(points) <= 1:
        return points
        
    # Find point furthest from center as starting point
    center = np.mean(points, axis=0)
    distances = np.sum((points - center) ** 2, axis=1)
    current_idx = np.argmax(distances)
    sorted_indices = [current_idx]
    
    # Precompute full distance matrix
    dist_matrix = cdist(points, points)
    
    # Mask for remaining points
    remaining = np.ones(len(points), dtype=bool)
    remaining[current_idx] = False
    
    # Iteratively find closest point
    while np.any(remaining):
        current_distances = dist_matrix[sorted_indices[-1]]
        # Mask visited points with infinite distance
        masked_distances = np.where(remaining, current_distances, np.inf)
        next_idx = np.argmin(masked_distances)
        sorted_indices.append(next_idx)
        remaining[next_idx] = False
    
    return points[sorted_indices]

def detect_corners_improved(points, min_angle=20, min_dist=15):
    """More efficient corner detection with vectorized operations"""
    if len(points) < 30:  # Need enough points for meaningful corners
        return [points]
        
    sorted_points = sort_points_vectorized(points)
    window_size, stride = 25, 4  # Increased stride for efficiency
    
    corners = []
    
    # Vectorized angle calculation
    for i in range(0, len(sorted_points) - window_size, stride):
        p1, p2, p3 = sorted_points[i], sorted_points[i + window_size//2], sorted_points[i + window_size]
        v1, v2 = p2 - p1, p3 - p2
        
        # Avoid division by zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2) 
        if norm_v1 < 1e-10 or norm_v2 < 1e-10:
            continue
            
        cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        idx = i + window_size//2
        if angle_deg > min_angle and (not corners or (idx - corners[-1]) >= min_dist):
            corners.append(idx)
    
    # Create segments from corners
    segments, start_idx = [], 0
    for corner_idx in corners:
        segment = sorted_points[start_idx:corner_idx]
        if len(segment) >= 5:  # Minimum points for a meaningful segment
            segments.append(segment)
        start_idx = corner_idx
    
    # Add final segment
    if len(sorted_points[start_idx:]) >= 5:
        segments.append(sorted_points[start_idx:])
    
    return segments if segments else [sorted_points]

def get_keyway_corners(mask):
    """Extract corners from keyway mask using contour approximation"""
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Try different epsilon values for approximation
    for eps_factor in [0.02, 0.01, 0.03, 0.05]:
        epsilon = eps_factor * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            break
    
    # Convert to [y, x] format and sort by angle
    corners = np.array([[p[0][1], p[0][0]] for p in approx])
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
    return corners[np.argsort(angles)]

def fit_line(points, is_horizontal=True):
    """Fit line to points using RANSAC"""
    if len(points) < 5:  # Need enough points for robust fitting
        return None
        
    # Select appropriate dimension as predictor variable
    x = points[:, 1 if is_horizontal else 0].reshape(-1, 1)
    y = points[:, 0 if is_horizontal else 1]
    
    try:
        ransac = RANSACRegressor(random_state=42, max_trials=100)
        ransac.fit(x, y)
        return (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)
    except:
        return None

def create_line_from_points(p1, p2):
    """Create line parameters from two points"""
    y1, x1 = p1
    y2, x2 = p2
    
    # Handle vertical lines
    dx = x2 - x1
    if abs(dx) < 1e-10:
        return (0, x1, False)  # Vertical line
    
    # Regular line
    slope = (y2 - y1) / dx
    intercept = y1 - slope * x1
    return (slope, intercept, abs(slope) < 1)  # Include "is_horizontal" flag

def process_boundary_segments(boundary_mask, min_cluster_size=10):
    """Extract and cluster boundary segments more efficiently"""
    # Create skeleton
    boundary_uint8 = boundary_mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.subtract(boundary_uint8, cv2.erode(boundary_uint8, kernel, iterations=1))
    
    # Get skeleton points
    skeleton_points = np.column_stack(np.where(skeleton > 0))
    if len(skeleton_points) < min_cluster_size:
        return []
    
    # Cluster points
    try:
        clustering = DBSCAN(eps=20, min_samples=5).fit(skeleton_points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Group points by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(skeleton_points[i])
        
        # Convert to numpy arrays
        clusters = [np.array(cluster) for cluster in clusters if len(cluster) >= min_cluster_size]
        
        # Find corners in each cluster
        segments = []
        for cluster in clusters:
            segments.extend(detect_corners_improved(cluster, min_angle=19.5, min_dist=15))
        
        return [seg for seg in segments if len(seg) >= min_cluster_size]
    except:
        return []

def match_boundary_lines(keyway_lines, boundary_segments, keyway_mask):
    """Match keyway lines to boundary segments with improved efficiency"""
    if not boundary_segments or not keyway_lines:
        return [None] * len(keyway_lines)
    
    matches = []
    
    # Calculate keyway center once
    keyway_points = np.column_stack(np.where(keyway_mask))
    if len(keyway_points) == 0:
        return [None] * len(keyway_lines)
    keyway_center = np.mean(keyway_points, axis=0)
    
    # Precompute boundary line fits
    boundary_lines = []
    for points in boundary_segments:
        # Try horizontal fit first, then vertical
        for is_horizontal in [True, False]:
            line_fit = fit_line(points, is_horizontal)
            if line_fit:
                boundary_lines.append(((line_fit[0], line_fit[1], is_horizontal), points))
                break
    
    # Match each keyway line to best boundary
    for keyway_line, keyway_points in keyway_lines:
        slope, intercept, is_horizontal = keyway_line
        best_match, best_score = None, -1
        
        # Calculate midpoint and outward vector
        midpoint = np.mean(keyway_points, axis=0)
        outward_vector = midpoint - keyway_center
        if np.linalg.norm(outward_vector) > 0:
            outward_vector = outward_vector / np.linalg.norm(outward_vector)
        
        # Find best matching boundary
        for boundary_line, boundary_points in boundary_lines:
            b_slope, b_intercept, b_is_horizontal = boundary_line
            
            # Skip if orientation doesn't match
            if is_horizontal != b_is_horizontal or abs(slope - b_slope) > 0.5:
                continue
                
            # Calculate vector to boundary midpoint
            boundary_midpoint = np.mean(boundary_points, axis=0)
            to_boundary_vector = boundary_midpoint - midpoint
            
            if np.linalg.norm(to_boundary_vector) > 0:
                to_boundary_vector = to_boundary_vector / np.linalg.norm(to_boundary_vector)
                
                # Score based on direction alignment and distance
                direction_score = np.dot(outward_vector, to_boundary_vector)
                if direction_score <= 0:
                    continue
                    
                # Calculate minimum distance efficiently
                distances = cdist([midpoint], boundary_points)[0]
                min_distance = np.min(distances)
                
                # Calculate boundary length
                length = (np.max(boundary_points[:, 1]) - np.min(boundary_points[:, 1])) if is_horizontal else (np.max(boundary_points[:, 0]) - np.min(boundary_points[:, 0]))
                
                # Final score
                score = (length / (1 + min_distance)) * direction_score
                
                if score > best_score:
                    best_score = score
                    best_match = (boundary_line, boundary_points)
        
        matches.append(best_match)
    
    return matches

def identify_middle_line(matched_lines, keyway_mask):
    """Find the middle line from matched boundary lines"""
    if not matched_lines or all(m is None for m in matched_lines):
        return None
        
    # Calculate keyway center
    y_indices, x_indices = np.where(keyway_mask)
    if len(y_indices) == 0:
        return None
        
    center_y, center_x = np.mean(y_indices), np.mean(x_indices)
    
    closest_dist, middle_idx = float('inf'), None
    
    # Find line closest to center
    for i, match in enumerate(matched_lines):
        if match is None:
            continue
            
        line, _ = match
        slope, intercept, is_horizontal = line
        
        # Calculate distance from center to line
        if is_horizontal:
            dist = abs(center_y - (slope * center_x + intercept))
        else:
            dist = abs(center_x - (slope * center_y + intercept))
        
        if dist < closest_dist:
            closest_dist = dist
            middle_idx = i
    
    return middle_idx

def compute_intersections(matched_lines, middle_idx, shape):
    """Calculate line intersections and canvas extensions"""
    if middle_idx is None or matched_lines[middle_idx] is None:
        return [], (0, 0, 0, 0)
        
    height, width = shape
    
    # Get middle line
    middle_match = matched_lines[middle_idx]
    middle_line, _ = middle_match
    middle_slope, middle_intercept, middle_is_horizontal = middle_line
    
    # Initialize
    intersections = []
    top_ext = right_ext = bottom_ext = left_ext = 0
    
    # Calculate intersections with other lines
    for i, match in enumerate(matched_lines):
        if match is None or i == middle_idx:
            continue
            
        line, _ = match
        slope, intercept, is_horizontal = line
        
        # Skip parallel lines
        if abs(middle_slope - slope) < 1e-6:
            continue
            
        # Calculate intersection point
        if middle_is_horizontal and is_horizontal:
            x = (intercept - middle_intercept) / (middle_slope - slope)
            y = middle_slope * x + middle_intercept
        elif not middle_is_horizontal and not is_horizontal:
            y = (intercept - middle_intercept) / (middle_slope - slope)
            x = middle_slope * y + middle_intercept
        elif middle_is_horizontal:
            y = (middle_slope * intercept + middle_intercept) / (1 - middle_slope * slope)
            x = slope * y + intercept
        else:
            x = (middle_slope * intercept + middle_intercept) / (1 - middle_slope * slope)
            y = slope * x + intercept
        
        # Only add valid intersections
        if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
            intersections.append((int(x), int(y)))
            
            # Calculate necessary canvas extensions
            if x < 0:
                left_ext = max(left_ext, int(-x + 20))
            elif x >= width:
                right_ext = max(right_ext, int(x - width + 20))
                
            if y < 0:
                top_ext = max(top_ext, int(-y + 20))
            elif y >= height:
                bottom_ext = max(bottom_ext, int(y - height + 20))
    
    return intersections, (top_ext, right_ext, bottom_ext, left_ext)

def create_visualization(masks, matched_lines, middle_idx, intersections, extensions):
    """Create visualization with boundary lines and intersections"""
    keyway_mask = masks.get(1, None)
    three_mask = masks.get(2, None)
    
    if keyway_mask is None:
        return None
        
    height, width = keyway_mask.shape
    
    # Apply extensions
    top_ext, right_ext, bottom_ext, left_ext = extensions
    
    new_height = height + top_ext + bottom_ext
    new_width = width + left_ext + right_ext
    
    # Create extended canvas
    canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Create extended masks
    extended_keyway = np.zeros((new_height, new_width), dtype=bool)
    extended_keyway[top_ext:top_ext+height, left_ext:left_ext+width] = keyway_mask
    
    extended_three = np.zeros((new_height, new_width), dtype=bool)
    if three_mask is not None:
        extended_three[top_ext:top_ext+height, left_ext:left_ext+width] = three_mask
    
    # Apply masks to canvas
    canvas[extended_keyway] = (0, 255, 0)  # Green for keyway
    canvas[extended_three] = (0, 0, 255)   # Blue for three
    
    # Line colors
    colors = [
        (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128)
    ]
    
    # Draw boundary lines
    for i, match in enumerate(matched_lines):
        if match is None:
            continue
            
        line, _ = match
        slope, intercept, is_horizontal = line
        
        color = colors[i % len(colors)]
        
        # Calculate line endpoints
        if is_horizontal:
            x1, x2 = 0, new_width
            y1 = int(slope * -left_ext + intercept) + top_ext
            y2 = int(slope * (width + right_ext) + intercept) + top_ext
        else:
            y1, y2 = 0, new_height
            x1 = int(slope * -top_ext + intercept) + left_ext
            x2 = int(slope * (height + bottom_ext) + intercept) + left_ext
            
        # Draw line
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        label = f"B{i}"
        if i == middle_idx:
            label += "*"  # Mark middle line
        cv2.putText(canvas, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw intersections
    for x, y in intersections:
        adj_x, adj_y = x + left_ext, y + top_ext
        cv2.circle(canvas, (adj_x, adj_y), 8, (255, 255, 255), -1)
        cv2.circle(canvas, (adj_x, adj_y), 6, (100, 100, 200), -1)
    
    return canvas

def process_single_frame(h5_file, frame_number, save_images=False):
    """Process a single frame with all optimizations"""
    # Load frame data
    frame_data = load_frame_data(h5_file, frame_number)
    if frame_data is None or 'masks' not in frame_data:
        return None
        
    masks = frame_data['masks']
    mask_shape = frame_data['shape']
    
    # Check for required masks
    if 0 not in masks or 1 not in masks:  # Need boundary and keyway
        return None
        
    boundary_mask = masks[0]
    keyway_mask = masks[1]
    
    # Get keyway corners
    keyway_corners = get_keyway_corners(keyway_mask)
    if keyway_corners is None or len(keyway_corners) < 3:
        return None
    
    # Create keyway lines
    keyway_lines = []
    for i in range(len(keyway_corners)):
        p1 = keyway_corners[i]
        p2 = keyway_corners[(i + 1) % len(keyway_corners)]
        keyway_lines.append((create_line_from_points(p1, p2), np.array([p1, p2])))
    
    # Process boundary segments
    boundary_segments = process_boundary_segments(boundary_mask)
    
    # Match boundary lines
    matched_lines = match_boundary_lines(keyway_lines, boundary_segments, keyway_mask)
    
    # Find middle line
    middle_idx = identify_middle_line(matched_lines, keyway_mask)
    
    # Compute intersections
    intersections, extensions = compute_intersections(matched_lines, middle_idx, mask_shape)
    
    # Create visualization
    result_image = create_visualization(masks, matched_lines, middle_idx, intersections, extensions)
    
    # Save if requested
    if save_images and result_image is not None:
        cv2.imwrite(f"frame_{frame_number}_result.png", result_image)
    
    return {
        'frame': frame_number,
        'result_image': result_image,
        'keyway_corners': keyway_corners,
        'matched_lines': matched_lines,
        'middle_idx': middle_idx,
        'intersections': intersections
    }

def process_batch(args):
    """Process a batch of frames - designed for multiprocessing"""
    batch, h5_file, save_images = args
    results = {}
    for frame in tqdm(batch, desc=f"Process {mp.current_process().name}", leave=False):
        result = process_single_frame(h5_file, frame, save_images)
        if result is not None:
            results[frame] = result
    return results

def process_frames_parallel(h5_file, frame_numbers, n_processes=None, save_images=False):
    """Process multiple frames in parallel"""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    # Split frame numbers into batches for each process
    batch_size = len(frame_numbers) // n_processes + 1
    batches = [frame_numbers[i:i+batch_size] for i in range(0, len(frame_numbers), batch_size)]
    
    # Prepare arguments for each batch
    batch_args = [(batch, h5_file, save_images) for batch in batches]
    
    # Create and start worker processes
    with mp.Pool(processes=n_processes) as pool:
        batch_results = pool.map(process_batch, batch_args)
    
    # Combine results
    all_results = {}
    for res in batch_results:
        all_results.update(res)
    
    return all_results

def create_video_from_results(results, output_path, fps=10, frame_shape=None):
    """Create video directly from processing results"""
    if not results:
        print("No results to create video from")
        return
    
    # Get frame numbers and sort
    frame_numbers = sorted(list(results.keys()))
    
    # Get frame shape from first result if not provided
    if frame_shape is None and results[frame_numbers[0]]['result_image'] is not None:
        frame_shape = results[frame_numbers[0]]['result_image'].shape
    
    # Fallback shape
    if frame_shape is None:
        frame_shape = (480, 640, 3)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_shape[1], frame_shape[0]))
    
    # Create blank frame
    blank_frame = np.zeros(frame_shape, dtype=np.uint8)
    
    # Write frames
    for frame_num in tqdm(range(min(frame_numbers), max(frame_numbers) + 1), desc="Creating video"):
        if frame_num in results and results[frame_num]['result_image'] is not None:
            writer.write(results[frame_num]['result_image'])
        else:
            writer.write(blank_frame)
    
    writer.release()
    print(f"Video saved to {output_path}")

def process_h5_to_video(h5_file_path, output_video_path, start_frame=0, end_frame=None, fps=10, skip_frames=1, n_processes=None):
    """Main function to process H5 file and create video"""
    # Get frame information
    print("Analyzing H5 file...")
    frame_info = get_frame_info(h5_file_path)
    
    if not frame_info['valid_frames']:
        print("No valid frames found in H5 file")
        return
    
    valid_frames = frame_info['valid_frames']
    
    # Determine frame range
    if end_frame is None or end_frame >= len(valid_frames):
        end_frame = len(valid_frames) - 1
    
    # Select frames to process
    frames_to_process = valid_frames[start_frame:end_frame+1:skip_frames]
    print(f"Processing {len(frames_to_process)} frames out of {len(valid_frames)} total frames")
    
    # Process frames in parallel
    start_time = time.time()
    results = process_frames_parallel(h5_file_path, frames_to_process, n_processes)
    processing_time = time.time() - start_time
    
    print(f"Processed {len(results)} frames in {processing_time:.2f} seconds")
    print(f"Average time per frame: {processing_time/len(frames_to_process):.4f} seconds")
    
    # Create video
    create_video_from_results(results, output_video_path, fps, 
                             frame_shape=frame_info['shape'] and (frame_info['shape'][0], frame_info['shape'][1], 3))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process H5 file with optimized boundary line detection')
    parser.add_argument('h5_file', type=str, help='Path to the H5 file')
    parser.add_argument('--output', type=str, default='optimized_boundary_lines.mp4', help='Output video path')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=None, help='End frame index')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for output video')
    parser.add_argument('--skip', type=int, default=1, help='Process every n-th frame')
    parser.add_argument('--processes', type=int, default=None, help='Number of parallel processes to use')
    
    args = parser.parse_args()
    
    process_h5_to_video(
        args.h5_file, 
        args.output, 
        start_frame=args.start, 
        end_frame=args.end, 
        fps=args.fps, 
        skip_frames=args.skip,
        n_processes=args.processes
    )