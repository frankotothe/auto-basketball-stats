import numpy as np
import h5py
import cv2
from pathlib import Path

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

def find_shooter(h5_file_path, shot_frame, max_lookback=200, distance_threshold=2):
    shot_frame = int(shot_frame)
    
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
            closest_point = None
            
            for key in frame_group.keys():
                if not key.startswith('player'):
                    continue
                    
                player_mask = get_mask_from_player_h5(frame_group, key)
                if player_mask is None or not np.any(player_mask):
                    continue
                    
                player_pixels = np.where(player_mask)
                player_points = np.column_stack((player_pixels[0], player_pixels[1]))
                
                distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
                player_min_distance = np.min(distances)
                
                if player_min_distance < min_distance:
                    min_distance = player_min_distance
                    closest_player = key
                    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    closest_point = (player_points[min_idx[1]][0], player_points[min_idx[1]][1])
            
            if min_distance <= distance_threshold:
                return frame_idx, closest_player, (ball_x, ball_y), closest_point
    
    return None, None, None, None

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
            
    except Exception:
        return None

def create_visualization_with_court(court_h5_path, player_h5_path, frame_num, shooter_id, output_path):
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
    
    except Exception:
        pass
    
    try:
        with h5py.File(player_h5_path, 'r') as player_h5:
            player_frame_key = f"frame_{frame_num:05d}"
            if player_frame_key in player_h5:
                frame_group = player_h5[player_frame_key]
                
                for obj_key in frame_group.keys():
                    mask = get_mask_from_player_h5(frame_group, obj_key)
                    if mask is None:
                        continue
                    
                    color = (0, 255, 0) if obj_key == shooter_id else (255, 255, 255)
                    
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
                    
                    obj_metadata = dict(frame_group[obj_key].attrs)
                    text_x = int(float(obj_metadata['centroid_x']))
                    text_y = int(float(obj_metadata['centroid_y']))
                    
                    cv2.putText(combined_canvas, obj_key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 0, 0), 2)
                    cv2.putText(combined_canvas, obj_key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 255, 255), 1)
    
    except Exception:
        pass
    
    cv2.imwrite(str(output_path), combined_canvas)

def main():
    fileName = "TestClip2"
    court_h5_path = f'courtTrackingOutputs/{fileName}.h5'
    player_h5_path = f'trackingOutputs/{fileName}.h5'
    shot_csv_path = f'madeShotCSV/{fileName}.csv'
    
    output_dir = Path('madeShots')
    output_dir.mkdir(exist_ok=True)
    
    shots = load_shot_detections(shot_csv_path)
    sequence_starts = find_sequence_starts(shots)
    
    for shot_frame in sequence_starts:
        print(f"Processing shot sequence starting at frame {shot_frame}")
        
        frame_num, shooter_id, ball_pos, player_pos = find_shooter(player_h5_path, shot_frame)
        
        if frame_num is not None:
            print(f"\nFound shooter {shooter_id} in frame {frame_num}")
            print(f"Ball position: {ball_pos}")
            print(f"Closest player pixel: {player_pos}")
            
            output_path = output_dir / f'made_shot_{frame_num:05d}.png'
            create_visualization_with_court(court_h5_path, player_h5_path, frame_num, shooter_id, output_path)
        else:
            output_path = output_dir / f'shot_no_shooter_{shot_frame:05d}.png'
            create_visualization_with_court(court_h5_path, player_h5_path, shot_frame, None, output_path)

if __name__ == "__main__":
    main()