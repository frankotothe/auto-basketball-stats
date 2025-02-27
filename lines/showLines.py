import h5py
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def rle_decode(rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    for start, end in rle:
        mask[start:end] = True
    return mask.reshape(shape)

def sort_points(points):
    sorted_indices = [np.lexsort((points[:, 1], points[:, 0]))[0]]
    remaining = set(range(len(points)))
    remaining.remove(sorted_indices[0])
    
    while remaining:
        last_point = points[sorted_indices[-1]]
        closest_idx = min(remaining, key=lambda idx: np.sum((points[idx] - last_point) ** 2))
        sorted_indices.append(closest_idx)
        remaining.remove(closest_idx)
    
    return points[sorted_indices]

def detect_corners(points, min_angle=20, min_dist=15):
    sorted_points = sort_points(points)
    window_size = 25
    stride = 2
    
    corners = []
    last_corner_idx = -min_dist
    all_angles = []
    
    for i in range(0, len(sorted_points) - window_size, stride):
        p1, p2, p3 = sorted_points[i], sorted_points[i + window_size//2], sorted_points[i + window_size]
        v1, v2 = p2 - p1, p3 - p2
        
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product < 1e-10:
            continue
            
        cos_angle = np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        all_angles.append(angle_deg)
        
        idx = i + window_size//2
        if angle_deg > min_angle and (idx - last_corner_idx) >= min_dist:
            corners.append(idx)
            last_corner_idx = idx
    
    segments = []
    start_idx = 0
    
    for corner_idx in corners:
        segment = sorted_points[start_idx:corner_idx]
        if len(segment) >= 5:
            segments.append(segment)
        start_idx = corner_idx
    
    last_segment = sorted_points[start_idx:]
    if len(last_segment) >= 5:
        segments.append(last_segment)
    
    return segments if segments else [sorted_points]

def fit_line(points, is_horizontal=True):
    if len(points) < 2:
        return None
        
    x = points[:, 1 if is_horizontal else 0].reshape(-1, 1)
    y = points[:, 0 if is_horizontal else 1]
    
    try:
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(x, y)
        return (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)
    except:
        return None

def get_keyway_corners(mask):
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    for eps_factor in [0.02, 0.01, 0.03, 0.05, 0.1]:
        epsilon = eps_factor * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            break
    
    corners = np.array([[p[0][1], p[0][0]] for p in approx])
    
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
    return corners[np.argsort(angles)]

def create_line_from_points(p1, p2):
    y1, x1 = p1
    y2, x2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) < 1e-10:
        return (0, x1, False)
    else:
        slope = dy / dx
        intercept = y1 - slope * x1
        return (slope, intercept, abs(slope) < 1)

def match_boundary_lines(keyway_lines, boundary_segments, keyway_mask, mask_shape):
    matches = []
    
    keyway_points = np.column_stack(np.where(keyway_mask))
    if len(keyway_points) == 0:
        return [None] * len(keyway_lines)
    keyway_center = np.mean(keyway_points, axis=0)
    
    for keyway_line, keyway_points in keyway_lines:
        slope, intercept, is_horizontal = keyway_line
        best_match = None
        best_score = -1
        
        p1, p2 = keyway_points
        midpoint = np.mean(keyway_points, axis=0)
        
        outward_vector = midpoint - keyway_center
        outward_vector = outward_vector / np.linalg.norm(outward_vector)
        
        for boundary_points in boundary_segments:
            if len(boundary_points) < 10:
                continue
                
            boundary_fit = fit_line(boundary_points, is_horizontal)
            if not boundary_fit:
                continue
                
            b_slope, b_intercept = boundary_fit
            boundary_line = (b_slope, b_intercept, is_horizontal)
            
            slope_diff = abs(slope - b_slope)
            if is_horizontal and slope_diff > 0.5:
                continue
            
            boundary_midpoint = np.mean(boundary_points, axis=0)
            
            to_boundary_vector = boundary_midpoint - midpoint
            
            if np.linalg.norm(to_boundary_vector) > 0:
                to_boundary_vector = to_boundary_vector / np.linalg.norm(to_boundary_vector)
            else:
                continue
            
            direction_score = np.dot(outward_vector, to_boundary_vector)
            
            if direction_score <= 0:
                continue
                
            distances = cdist(keyway_points, boundary_points)
            distance = np.min(distances)
            
            length = np.max(boundary_points[:, 1]) - np.min(boundary_points[:, 1]) if is_horizontal else np.max(boundary_points[:, 0]) - np.min(boundary_points[:, 0])
            
            score = (length / (1 + distance)) * direction_score
            
            if score > best_score:
                best_score = score
                best_match = (boundary_line, boundary_points)
        
        matches.append(best_match)
    
    return matches

def identify_middle_line(matched_boundary_lines, keyway_mask):
    y_indices, x_indices = np.where(keyway_mask)
    center_y = np.mean(y_indices)
    center_x = np.mean(x_indices)
    
    closest_dist = float('inf')
    middle_line_idx = None
    
    for i, match in enumerate(matched_boundary_lines):
        if match is None:
            continue
            
        line, points = match
        slope, intercept, is_horizontal = line
        
        if is_horizontal:
            dist = abs(center_y - (slope * center_x + intercept))
        else:
            dist = abs(center_x - (slope * center_y + intercept))
        
        if dist < closest_dist:
            closest_dist = dist
            middle_line_idx = i
    
    return middle_line_idx

def compute_extended_boundaries(matched_boundary_lines, middle_line_idx, mask_shape):
    height, width = mask_shape
    top_ext = right_ext = bottom_ext = left_ext = 0
    
    middle_match = matched_boundary_lines[middle_line_idx]
    middle_slope, middle_intercept, middle_is_horizontal = middle_match[0]
    
    intersections = []
    
    for i, match in enumerate(matched_boundary_lines):
        if match is None or i == middle_line_idx:
            continue
            
        line, points = match
        slope, intercept, is_horizontal = line
        
        slope_diff = abs(middle_slope - slope)
        if slope_diff < 1e-6:
            continue
            
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
        
        if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
            intersections.append((int(x), int(y)))
            
            if x < 0:
                left_ext = max(left_ext, int(-x + 50))
            elif x >= width:
                right_ext = max(right_ext, int(x - width + 50))
                
            if y < 0:
                top_ext = max(top_ext, int(-y + 50))
            elif y >= height:
                bottom_ext = max(bottom_ext, int(y - height + 50))
    
    return intersections, (top_ext, right_ext, bottom_ext, left_ext)

def extend_canvas_and_lines(canvas, extension, intersections, matched_boundary_lines, keyway_mask, three_mask):
    top_ext, right_ext, bottom_ext, left_ext = extension
    
    orig_height, orig_width = canvas.shape[:2]
    
    new_height = orig_height + top_ext + bottom_ext
    new_width = orig_width + left_ext + right_ext
    extended_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    extended_canvas[top_ext:top_ext+orig_height, left_ext:left_ext+orig_width] = canvas
    
    extended_keyway = np.zeros((new_height, new_width), dtype=bool)
    extended_keyway[top_ext:top_ext+orig_height, left_ext:left_ext+orig_width] = keyway_mask

    extended_three = np.zeros((new_height, new_width), dtype=bool)
    extended_three[top_ext:top_ext+orig_height, left_ext:left_ext+orig_width] = three_mask
    
    extended_canvas[extended_keyway] = (0, 255, 0)
    extended_canvas[extended_three] = (0, 0, 255)
    
    colors = [
        (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
        (0, 128, 128), (255, 165, 0)
    ]
    
    for i, match in enumerate(matched_boundary_lines):
        if match is None:
            continue
            
        line, points = match
        slope, intercept, is_horizontal = line
        
        color = colors[i % len(colors)]
        
        if is_horizontal:
            x1 = -left_ext
            x2 = orig_width + right_ext
            y1 = int(slope * x1 + intercept) + top_ext
            y2 = int(slope * x2 + intercept) + top_ext
        else:
            y1 = -top_ext
            y2 = orig_height + bottom_ext
            x1 = int(slope * y1 + intercept) + left_ext
            x2 = int(slope * y2 + intercept) + left_ext
        
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        cv2.putText(extended_canvas, f"B{i}", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    for x, y in intersections:
        adj_x = x + left_ext
        adj_y = y + top_ext
        
        cv2.circle(extended_canvas, (adj_x, adj_y), 8, (255, 255, 255), -1)
        cv2.circle(extended_canvas, (adj_x, adj_y), 6, (100, 100, 200), -1)
    
    return extended_canvas

def find_boundary_lines(h5_file_path, frame_number=180):
    with h5py.File(h5_file_path, 'r') as f:
        frame_key = f'frame_{frame_number}'
        if frame_key not in f['frames']:
            print(f"Frame {frame_number} not found")
            return None
            
        frame = f['frames'][frame_key]
        
        detection_keys = list(frame.keys())
        if not detection_keys:
            print("No detections found in this frame")
            return None
            
        first_detection = frame[detection_keys[0]]
        mask_shape = tuple(first_detection['rle'].attrs['shape'])
        
        boundary_mask = np.zeros(mask_shape, dtype=bool)
        keyway_mask = np.zeros(mask_shape, dtype=bool)
        three_mask = np.zeros(mask_shape, dtype=bool)
        boundary_count = 0
        
        for detection_key in frame.keys():
            detection = frame[detection_key]
            class_id = detection.attrs['class_id']
            
            rle_data = detection['rle'][:]
            mask = rle_decode(rle_data, mask_shape)
            
            if class_id == 0:  # Boundary
                boundary_count += 1
                boundary_mask |= mask
            elif class_id == 1:  # Keyway
                keyway_mask |= mask
            elif class_id == 2:  # Three
                three_mask |= mask
        
        print(f"Number of boundary objects detected: {boundary_count}")
        
        keyway_corners = get_keyway_corners(keyway_mask)
        if keyway_corners is None or len(keyway_corners) < 3:
            print("Failed to extract keyway corners")
            return None
        
        print(f"Found {len(keyway_corners)} keyway corners")
        
        keyway_lines = []
        for i in range(len(keyway_corners)):
            p1 = keyway_corners[i]
            p2 = keyway_corners[(i + 1) % len(keyway_corners)]
            line = create_line_from_points(p1, p2)
            keyway_lines.append((line, np.array([p1, p2])))
        
        boundary_uint8 = boundary_mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(boundary_uint8, kernel, iterations=1)
        skeleton = cv2.subtract(boundary_uint8, eroded)
        
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) < 10:
            print("Not enough boundary points found")
            return None
        
        clustering = DBSCAN(eps=20, min_samples=5).fit(skeleton_points)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Number of boundary segments detected: {n_clusters}")
        
        clustered_points = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            if label != -1:  # Skip noise points
                clustered_points[label].append(skeleton_points[i])
        
        for i in range(n_clusters):
            clustered_points[i] = np.array(clustered_points[i])
        
        boundary_segments = []
        for points in clustered_points:
            if len(points) >= 10:
                sub_segments = detect_corners(points, min_angle=19.5, min_dist=15)
                boundary_segments.extend([seg for seg in sub_segments if len(seg) >= 10])
        
        print(f"Total boundary segments after corner splitting: {len(boundary_segments)}")
        
        boundary_canvas = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
        boundary_canvas[keyway_mask] = (0, 255, 0)  # Green for keyway
        
        for corner in keyway_corners:
            cv2.circle(boundary_canvas, (corner[1], corner[0]), 5, (255, 255, 255), -1)
        
        for line, points in keyway_lines:
            p1, p2 = points
            cv2.line(boundary_canvas, (p1[1], p1[0]), (p2[1], p2[0]), (255, 255, 255), 2)
        
        colors = [
            (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
            (0, 128, 128), (255, 165, 0), (255, 192, 203), (165, 42, 42),
            (240, 230, 140), (70, 130, 180)
        ]
        
        while len(colors) < len(boundary_segments):
            new_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            colors.append(new_color)
        
        for i, points in enumerate(boundary_segments):
            color = colors[i % len(colors)]
            
            for point in points:
                boundary_canvas[point[0], point[1]] = color
            
            for is_horizontal in [True, False]:
                line_fit = fit_line(points, is_horizontal)
                if line_fit is not None:
                    slope, intercept = line_fit
                    
                    if is_horizontal:
                        min_x = np.min(points[:, 1])
                        max_x = np.max(points[:, 1])
                        y1 = int(slope * min_x + intercept)
                        y2 = int(slope * max_x + intercept)
                        x1, x2 = min_x, max_x
                    else:
                        min_y = np.min(points[:, 0])
                        max_y = np.max(points[:, 0])
                        x1 = int(slope * min_y + intercept)
                        x2 = int(slope * max_y + intercept)
                        y1, y2 = min_y, max_y
                    
                    cv2.line(boundary_canvas, (x1, y1), (x2, y2), color, 2)
                    
                    text_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.putText(boundary_canvas, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    break
        
        legend_canvas = np.zeros((200, mask_shape[1], 3), dtype=np.uint8)
        start_y = 20
        for i in range(min(len(boundary_segments), len(colors))):
            color = colors[i % len(colors)]
            cv2.putText(legend_canvas, f"Segment {i}", (10, start_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.line(legend_canvas, (150, start_y - 5), (200, start_y - 5), color, 3)
            start_y += 15
        
        matched_boundary_lines = match_boundary_lines(keyway_lines, boundary_segments, keyway_mask, mask_shape)
        
        match_canvas = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
        match_canvas[keyway_mask] = (0, 255, 0)  # Green for keyway
        
        for i, (line, points) in enumerate(keyway_lines):
            p1, p2 = points
            cv2.line(match_canvas, (p1[1], p1[0]), (p2[1], p2[0]), (255, 255, 255), 2)
            mid_x = int((p1[1] + p2[1]) / 2)
            mid_y = int((p1[0] + p2[0]) / 2)
            cv2.putText(match_canvas, f"K{i}", (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        print("\nKeyway to Boundary Line Matches:")
        for i, match in enumerate(matched_boundary_lines):
            if match is None:
                print(f"  Keyway line {i}: No match found")
                continue
                
            line, points = match
            slope, intercept, is_horizontal = line
            
            color = colors[i % len(colors)]
            
            if is_horizontal:
                min_x = np.min(points[:, 1])
                max_x = np.max(points[:, 1])
                y1 = int(slope * min_x + intercept)
                y2 = int(slope * max_x + intercept)
                x1, x2 = min_x, max_x
            else:
                min_y = np.min(points[:, 0])
                max_y = np.max(points[:, 0])
                x1 = int(slope * min_y + intercept)
                x2 = int(slope * max_y + intercept)
                y1, y2 = min_y, max_y
            
            cv2.line(match_canvas, (x1, y1), (x2, y2), color, 3)
            
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            cv2.putText(match_canvas, f"B{i}", (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            segment_idx = next((idx for idx, seg in enumerate(boundary_segments) 
                              if np.array_equal(seg, points)), -1)
            print(f"  Keyway line {i} -> Boundary segment {segment_idx}")
        
        boundary_canvas[three_mask] = (0, 0, 255)  # Blue for three
        
        middle_line_idx = identify_middle_line(matched_boundary_lines, keyway_mask)
        if middle_line_idx is not None:
            print(f"\nIdentified middle line: B{middle_line_idx}")
            
            intersections, extension = compute_extended_boundaries(
                matched_boundary_lines, middle_line_idx, mask_shape
            )
            print(f"Found {len(intersections)} intersection points")
            print(f"Canvas extension needed: {extension}")
            
            extended_canvas = extend_canvas_and_lines(
                match_canvas, extension, intersections, matched_boundary_lines, keyway_mask, three_mask
            )
            
            cv2.imshow(f"Frame {frame_number} Extended Lines", extended_canvas)
            cv2.imwrite(f"frame_{frame_number}_extended_lines.png", extended_canvas)
        else:
            print("Could not identify a middle line")
            extended_canvas = match_canvas.copy()
        
        combined_canvas = np.vstack([boundary_canvas, legend_canvas, match_canvas])
        
        cv2.imshow(f"Frame {frame_number} Boundary Analysis", combined_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite(f"frame_{frame_number}_boundary_segments.png", boundary_canvas)
        cv2.imwrite(f"frame_{frame_number}_matches.png", match_canvas)
        cv2.imwrite(f"frame_{frame_number}_combined.png", combined_canvas)
        
        print(f"\nSummary:")
        print(f"- Found {boundary_count} raw boundary objects")
        print(f"- Extracted {n_clusters} boundary segments using DBSCAN")
        print(f"- After splitting at corners, there are {len(boundary_segments)} total segments")
        print(f"- Found {len(keyway_corners)} keyway corners forming {len(keyway_lines)} edges")
        print(f"- Successfully matched {sum(1 for m in matched_boundary_lines if m is not None)}/{len(keyway_lines)} keyway lines")
        if middle_line_idx is not None:
            print(f"- Identified B{middle_line_idx} as the middle line")
            print(f"- Found {len(intersections)} intersection points between extended lines")
        
        return {
            'keyway_corners': keyway_corners,
            'keyway_lines': keyway_lines,
            'boundary_segments': boundary_segments,
            'boundary_lines': matched_boundary_lines,
            'middle_line_idx': middle_line_idx,
            'intersections': intersections if middle_line_idx is not None else [],
            'canvas': combined_canvas,
            'extended_canvas': extended_canvas
        }

if __name__ == "__main__":
    results = find_boundary_lines('output.h5', frame_number=40)