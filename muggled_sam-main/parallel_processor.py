import os
import cv2
import h5py
import shutil
import tempfile
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import json
import argparse

def split_video_segments(video_path: str, 
                          split_points: Optional[List[int]] = None, 
                          num_segments: int = 4) -> List[Tuple[str, int, int]]:
    """
    Split the input video into segments
    
    Args:
        video_path: Path to the input video
        split_points: List of frame indices to split at (if provided, num_segments is ignored)
        num_segments: Number of equal segments to split into (if split_points not provided)
    
    Returns:
        List of (segment_path, start_frame, end_frame) tuples
    """
    # Create temp directory for segments
    temp_dir = tempfile.mkdtemp(prefix="video_segments_")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine segment boundaries
    if split_points:
        # Ensure split points are sorted and within range
        split_points = sorted([p for p in split_points if 0 < p < total_frames])
        boundaries = [0] + split_points + [total_frames]
    else:
        # Create evenly spaced segments
        frames_per_segment = total_frames // num_segments
        boundaries = [i * frames_per_segment for i in range(num_segments)]
        boundaries.append(total_frames)  # Add the end boundary
    
    # Create segment files
    segments = []
    
    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]
        segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create writer for this segment
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
        
        # Extract frames for this segment
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        
        writer.release()
        segments.append((segment_path, start_frame, end_frame))
    
    cap.release()
    return segments

def process_segment(segment_tuple: Tuple[str, int, int], 
                   unique_id_start: int,
                   script_path: str) -> Dict:
    """
    Process a single video segment using the original script
    
    Args:
        segment_tuple: (segment_path, start_frame, end_frame)
        unique_id_start: Starting unique ID for this segment
        script_path: Path to the original processing script
    
    Returns:
        Dictionary with segment results and metadata
    """
    segment_path, start_frame, end_frame = segment_tuple
    segment_dir = os.path.dirname(segment_path)
    segment_name = os.path.basename(segment_path).split('.')[0]
    
    # Create output paths
    output_original = os.path.join(segment_dir, f"{segment_name}_original.mp4")
    output_masks = os.path.join(segment_dir, f"{segment_name}_masks.mp4")
    output_labels = os.path.join(segment_dir, f"{segment_name}_labels.mp4")
    tracking_h5_path = os.path.join(segment_dir, f"{segment_name}_tracking.h5")
    
    # Create a configuration file for this segment
    config = {
        "video_path": segment_path,
        "output_original": output_original,
        "output_masks": output_masks,
        "output_labels": output_labels,
        "tracking_h5_path": tracking_h5_path,
        "unique_id_start": unique_id_start
    }
    
    config_path = os.path.join(segment_dir, f"{segment_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Run the original script with modified config
    subprocess.run(["python", script_path, "--config", config_path], check=True)
    
    # Return information about the processed segment
    return {
        "segment_path": segment_path,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "output_original": output_original,
        "output_masks": output_masks,
        "output_labels": output_labels,
        "tracking_h5_path": tracking_h5_path,
        "unique_id_start": unique_id_start
    }

def combine_h5_files(segment_results: List[Dict], output_h5_path: str):
    """
    Merge all segment H5 files into a single file
    
    Args:
        segment_results: List of segment result dictionaries
        output_h5_path: Path to the output H5 file
    """
    # Create new H5 file
    with h5py.File(output_h5_path, 'w') as final_h5:
        # Copy metadata from first segment
        first_segment_h5 = h5py.File(segment_results[0]["tracking_h5_path"], 'r')
        for key, value in first_segment_h5.attrs.items():
            final_h5.attrs[key] = value
        first_segment_h5.close()
        
        # Process each segment
        for segment in segment_results:
            with h5py.File(segment["tracking_h5_path"], 'r') as segment_h5:
                # Adjust frame numbers based on segment start frame
                offset = segment["start_frame"]
                
                # Copy and rename groups
                for frame_name in segment_h5.keys():
                    # Extract the frame number part
                    frame_num = int(frame_name.split('_')[1])
                    # Create the new frame name with offset
                    new_frame_name = f"frame_{frame_num + offset:05d}"
                    
                    # Copy the group with the new name
                    segment_h5.copy(frame_name, final_h5, name=new_frame_name)

def combine_videos(segment_results: List[Dict], output_prefix: str):
    """
    Concatenate video segments into final outputs
    
    Args:
        segment_results: List of segment result dictionaries
        output_prefix: Prefix for the output files
    """
    # Sort segments by start frame
    segment_results.sort(key=lambda x: x["start_frame"])
    
    # Prepare lists for each video type
    original_list = os.path.join(tempfile.gettempdir(), "original_list.txt")
    masks_list = os.path.join(tempfile.gettempdir(), "masks_list.txt")
    labels_list = os.path.join(tempfile.gettempdir(), "labels_list.txt")
    
    # Create list files for ffmpeg concat
    with open(original_list, 'w') as f_orig, \
         open(masks_list, 'w') as f_masks, \
         open(labels_list, 'w') as f_labels:
        
        for segment in segment_results:
            f_orig.write(f"file '{os.path.abspath(segment['output_original'])}'\n")
            f_masks.write(f"file '{os.path.abspath(segment['output_masks'])}'\n")
            f_labels.write(f"file '{os.path.abspath(segment['output_labels'])}'\n")
    
    # Concatenate using ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", original_list, "-c", "copy", f"{output_prefix}_original.mp4"
    ], check=True)
    
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", masks_list, "-c", "copy", f"{output_prefix}_masks.mp4"
    ], check=True)
    
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", labels_list, "-c", "copy", f"{output_prefix}_labels.mp4"
    ], check=True)
    
    # Clean up list files
    for f in [original_list, masks_list, labels_list]:
        if os.path.exists(f):
            os.remove(f)

def main(video_path: str, 
         script_path: str, 
         output_prefix: str = "final_output",
         split_points: Optional[List[int]] = None,
         num_segments: int = 4,
         max_workers: int = None) -> None:
    """
    Main function to process video in segments
    
    Args:
        video_path: Path to the input video
        script_path: Path to the original processing script
        output_prefix: Prefix for the output files
        split_points: List of frame indices to split at
        num_segments: Number of segments (used if split_points not provided)
        max_workers: Maximum number of parallel workers
    """
    try:
        # Split video into segments
        segments = split_video_segments(video_path, split_points, num_segments)
        
        # Process segments in parallel
        segment_results = []
        unique_id_counter = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all segment jobs
            futures = {}
            for segment in segments:
                future = executor.submit(
                    process_segment, segment, unique_id_counter, script_path
                )
                futures[future] = segment
                # Estimate unique IDs needed for next segment
                _, start, end = segment
                unique_id_counter += (end - start) // 5  # Conservative estimate for unique IDs
            
            # Process results as they complete
            for future in as_completed(futures):
                segment_result = future.result()
                segment_results.append(segment_result)
        
        # Combine results
        combine_videos(segment_results, output_prefix)
        combine_h5_files(segment_results, f"{output_prefix}_tracking.h5")
        
        print(f"Processing complete. Final outputs:")
        print(f"  - {output_prefix}_original.mp4")
        print(f"  - {output_prefix}_masks.mp4")
        print(f"  - {output_prefix}_labels.mp4")
        print(f"  - {output_prefix}_tracking.h5")
        
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        raise
    finally:
        # Clean up temp files (comment out during testing if needed)
        for segment_path, _, _ in segments:
            temp_dir = os.path.dirname(segment_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video in parallel segments")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("script_path", help="Path to original processing script")
    parser.add_argument("--output", default="final_output", help="Prefix for output files")
    parser.add_argument("--split_points", type=int, nargs="+", help="Frame indices to split at")
    parser.add_argument("--num_segments", type=int, default=4, help="Number of segments to split into")
    parser.add_argument("--max_workers", type=int, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    main(
        args.video_path,
        args.script_path,
        args.output,
        args.split_points,
        args.num_segments,
        args.max_workers
    )