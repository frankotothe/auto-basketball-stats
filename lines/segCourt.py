import cv2
import h5py
import numpy as np
from ultralytics import YOLO

def rle_encode(mask):
    """
    Encode binary mask using run-length encoding.
    Returns a list of [start, length] pairs.
    """
    # Find transitions between 0 and 1
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    
    # Pairs of (start, length)
    runs = runs.reshape(-1, 2)
    return runs.tolist()

def process_video(model_path, video_path, output_path=None, h5_output_path=None):
    """
    Run YOLO segmentation on a video file at 10 FPS with reduced mask overlap
    Args:
        model_path: Path to the YOLO model
        video_path: Path to input video
        output_path: Optional path to save output video
        h5_output_path: Optional path to save mask data in HDF5 format with RLE
    """
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, round(original_fps / 10))  # Process every nth frame to achieve 10 FPS
    print(f"Processing at native resolution: {frame_width}x{frame_height}, Target FPS: 10")
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (frame_width, frame_height))
    
    # Initialize HDF5 file if output path is provided
    h5_file = None
    if h5_output_path:
        h5_file = h5py.File(h5_output_path, 'w')
        # Create groups for different data types
        h5_file.create_group('frames')
        h5_file.create_group('classes')
        h5_file.create_group('confidences')
        h5_file.create_group('masks_rle')
        
        # Store metadata
        h5_file.attrs['frame_width'] = frame_width
        h5_file.attrs['frame_height'] = frame_height
        h5_file.attrs['original_fps'] = original_fps
        h5_file.attrs['target_fps'] = 10

    frame_count = 0
    processed_count = 0
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame to maintain 10 FPS
        if frame_count % frame_interval == 0:
            results = model.predict(
                source=frame,
                conf=0.25,
                show=False,
                stream=True,
                verbose=False,
                iou=0.5  # Adjust IoU threshold to reduce overlapping masks
            )
            
            # Process results
            for r in results:
                annotated_frame = r.plot(line_width=1, font_size=0.5)  # Thinner outlines and text
                
                # Display the frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                
                # Save frame if output path was provided
                if out:
                    out.write(annotated_frame)
                
                # Write mask data to HDF5 if provided
                if h5_file:
                    frame_group = h5_file['frames'].create_group(f'frame_{processed_count}')
                    frame_group.attrs['timestamp'] = frame_count
                    
                    if r.masks is not None:
                        for i, (box, mask) in enumerate(zip(r.boxes, r.masks.data)):
                            # Convert mask to binary numpy array
                            binary_mask = mask.cpu().numpy().astype(bool)
                            
                            # Run-length encode the mask
                            rle_data = rle_encode(binary_mask)
                            
                            # Store data in HDF5
                            mask_group = frame_group.create_group(f'detection_{i}')
                            mask_group.attrs['class_id'] = int(box.cls.item())
                            mask_group.attrs['confidence'] = float(box.conf.item())
                            
                            # Store RLE data
                            rle_dataset = mask_group.create_dataset('rle', data=np.array(rle_data))
                            rle_dataset.attrs['shape'] = binary_mask.shape
            
            processed_count += 1
        
        frame_count += 1
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    if h5_file:
        h5_file.close()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = "models/CourtYolo.pt"
    video_path = "1min.mp4"
    output_path = "output.mp4"  # Optional
    h5_output_path = "output.h5"  # Optional
    
    try:
        process_video(model_path, video_path, output_path, h5_output_path)
    except Exception as e:
        print(f"Error occurred: {e}")