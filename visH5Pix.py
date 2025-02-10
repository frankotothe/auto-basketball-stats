import numpy as np
import h5py
import cv2
from pathlib import Path
import tqdm

def get_mask_from_h5(frame_group, object_key):
    """Extract binary mask from H5 frame group for given object."""
    try:
        obj = frame_group[object_key]
        rle_data = obj['mask_rle'][:]  # Will be uint32 array of shape (N,2)
        shape = tuple(obj['mask_shape'][:])
        
        # Create mask from RLE data
        mask = np.zeros(shape[0] * shape[1], dtype=bool)
        for start, length in rle_data:
            mask[start:start+length] = True
        return mask.reshape(shape)
    except KeyError:
        # Object doesn't exist in this frame - this is normal
        return None
def find_closest_player(frame_group):
    """Find the player closest to the ball based on pixel distances."""
    if 'ball' not in frame_group:
        return None, None, None
        
    ball_mask = get_mask_from_h5(frame_group, 'ball')
    if ball_mask is None:
        return None, None, None
        
    # Get ball centroid
    ball_metadata = dict(frame_group['ball'].attrs)
    ball_x = int(float(np.float64(ball_metadata.get('centroid_x', 0))))
    ball_y = int(float(np.float64(ball_metadata.get('centroid_y', 0))))
    
    # Get ball pixels instead of just using centroid
    ball_pixels = np.where(ball_mask)
    if len(ball_pixels[0]) == 0:
        return None, None, None
        
    # Use ball's actual pixels for comparison
    ball_points = np.column_stack((ball_pixels[0], ball_pixels[1]))
    
    min_distance = float('inf')
    closest_player = None
    
    for key in frame_group.keys():
        if not key.startswith('player'):
            continue
            
        player_mask = get_mask_from_h5(frame_group, key)
        if player_mask is None or not np.any(player_mask):
            continue
            
        player_pixels = np.where(player_mask)
        player_points = np.column_stack((player_pixels[0], player_pixels[1]))
        
        # Calculate minimum distance between any ball pixel and any player pixel
        # Using broadcasting to calculate all distances at once
        distances = np.sqrt(np.sum((ball_points[:, np.newaxis] - player_points) ** 2, axis=2))
        player_min_distance = np.min(distances)
        
        if player_min_distance < min_distance:
            min_distance = player_min_distance
            closest_player = key
    
    return closest_player, min_distance, (ball_x, ball_y)

def create_visualization(h5_path, output_path, fps=30):
    """Create MP4 visualization of all frames."""
    with h5py.File(h5_path, 'r') as h5_file:
        # Get first frame to determine dimensions
        first_frame_key = sorted(h5_file.keys())[0]
        first_frame = h5_file[first_frame_key]
        first_obj = next(iter(first_frame.values()))
        height, width = first_obj['mask_shape'][:]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_keys = sorted(h5_file.keys())
        print(f"Processing {len(frame_keys)} frames...")
        
        for frame_key in tqdm.tqdm(frame_keys):
            frame_group = h5_file[frame_key]
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate closest player for this frame
            closest_player, min_distance, ball_pos = find_closest_player(frame_group)
            
            # Draw each object
            for obj_key in frame_group.keys():
                mask = get_mask_from_h5(frame_group, obj_key)
                if mask is None:
                    continue
                
                # Choose color based on object type
                if obj_key == 'ball':
                    color = (0, 165, 255)  # Orange for ball
                elif obj_key.startswith('player'):
                    if obj_key == closest_player:
                        color = (0, 255, 0)  # Green for closest player
                    else:
                        color = (255, 255, 255)  # White for other players
                else:
                    continue  # Skip non-player, non-ball objects
                
                # Apply mask
                for c in range(3):
                    image[:, :, c][mask] = color[c]
                
                # Draw centroid and label
                obj_metadata = dict(frame_group[obj_key].attrs)
                centroid_x = int(float(np.float64(obj_metadata['centroid_x'])))
                centroid_y = int(float(np.float64(obj_metadata['centroid_y'])))
                
                # Draw centroid point
                cv2.circle(image, (centroid_x, centroid_y), 3, (0, 0, 255), -1)
                
                # Add object ID
                cv2.putText(image, obj_key, (centroid_x + 5, centroid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(image, obj_key, (centroid_x + 5, centroid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add distance information
            if closest_player and min_distance is not None:
                distance_text = f"Closest player: {closest_player} - Distance: {min_distance:.1f} pixels"
                cv2.putText(image, distance_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(image, distance_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add frame number
            frame_num = int(frame_key.split('_')[1])
            cv2.putText(image, f"Frame: {frame_num}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, f"Frame: {frame_num}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(image)
        
        out.release()
        print(f"Video saved to {output_path}")

def main():
    h5_path = 'trackingOutputs/object_tracking.h5'
    output_path = Path('tracking_visualization.mp4')
    create_visualization(h5_path, output_path)

if __name__ == "__main__":
    main()