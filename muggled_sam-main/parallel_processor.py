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
from tqdm import tqdm
import sys
from threading import Lock
import time

# Global progress bars dictionary and lock
progress_bars = {}
progress_lock = Lock()

def create_progress_bar(segment_id: int, total_frames: int) -> tqdm:
    """Create a progress bar for a segment"""
    with progress_lock:
        progress_bars[segment_id] = tqdm(
            total=total_frames,
            desc=f"Segment {segment_id:02d}",
            position=segment_id,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        return progress_bars[segment_id]

def update_progress(segment_id: int, frame_num: int):
    """Update progress for a specific segment"""
    with progress_lock:
        if segment_id in progress_bars:
            progress_bars[segment_id].n = frame_num
            progress_bars[segment_id].refresh()

def process_segment(segment_tuple: Tuple[str, int, int], 
                   unique_id_start: int,
                   script_path: str,
                   segment_id: int) -> Dict:
    """Process a single video segment using the original script"""
    segment_path, start_frame, end_frame = segment_tuple
    segment_dir = os.path.dirname(segment_path)
    segment_name = os.path.basename(segment_path).split('.')[0]
    total_frames = end_frame - start_frame
    
    # Create progress bar for this segment
    pbar = create_progress_bar(segment_id, total_frames)
    
    try:
        # Create output paths
        output_paths = {
            "output_original": os.path.join(segment_dir, f"{segment_name}_original.mp4"),
            "output_masks": os.path.join(segment_dir, f"{segment_name}_masks.mp4"),
            "output_labels": os.path.join(segment_dir, f"{segment_name}_labels.mp4"),
            "tracking_h5_path": os.path.join(segment_dir, f"{segment_name}_tracking.h5"),
            "progress_file": os.path.join(segment_dir, f"{segment_name}_progress.txt")
        }
        
        # Create config file
        config = {
            "video_path": segment_path,
            **output_paths,
            "unique_id_start": unique_id_start
        }
        
        config_path = os.path.join(segment_dir, f"{segment_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Start processing
        process = subprocess.Popen(
            ["python", script_path, "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress file
        last_frame = 0
        error_lines = []
        
        while process.poll() is None:
            # Check progress file
            if os.path.exists(config["progress_file"]):
                try:
                    with open(config["progress_file"], 'r') as f:
                        current_frame = int(f.read().strip())
                        if current_frame > last_frame:
                            update_progress(segment_id, current_frame)
                            last_frame = current_frame
                except (ValueError, IOError):
                    pass
            
            # Check for error output
            if process.stderr:
                line = process.stderr.readline()
                if line:
                    error_lines.append(line.strip())
            
            time.sleep(0.1)  # Prevent busy waiting
        
        # Final communication
        stdout, stderr = process.communicate()
        if stderr:
            error_lines.extend(stderr.splitlines())
        
        if process.returncode != 0:
            error_msg = "\n".join(error_lines) if error_lines else "Unknown error"
            with progress_lock:
                print(f"\nError in segment {segment_id}:", file=sys.stderr)
                print(error_msg, file=sys.stderr)
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                output=stdout,
                stderr=stderr
            )
        
        pbar.close()
        
        return {
            "segment_path": segment_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            **output_paths,
            "unique_id_start": unique_id_start
        }
        
    except Exception as e:
        pbar.close()
        raise e
    finally:
        # Cleanup progress file
        if os.path.exists(config["progress_file"]):
            try:
                os.remove(config["progress_file"])
            except OSError:
                pass

def main(video_path: str, 
         script_path: str, 
         output_prefix: str = "final_output",
         split_points: Optional[List[int]] = None,
         num_segments: int = 4,
         max_workers: int = None) -> None:
    """Main function to process video in segments with progress bars"""
    try:
        # Split video into segments
        segments = split_video_segments(video_path, split_points, num_segments)
        
        # Process segments in parallel
        segment_results = []
        unique_id_counter = 0
        
        print(f"\nProcessing {len(segments)} segments in parallel:")
        print("\nProgress bars for each segment:")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all segment jobs
            futures = {}
            for i, segment in enumerate(segments):
                future = executor.submit(
                    process_segment, 
                    segment, 
                    unique_id_counter, 
                    script_path,
                    i  # Pass segment ID for progress bar
                )
                futures[future] = segment
                _, start, end = segment
                unique_id_counter += (end - start) // 5
            
            # Process results as they complete
            failed_segments = []
            for future in as_completed(futures):
                try:
                    segment_result = future.result()
                    segment_results.append(segment_result)
                except Exception as e:
                    segment = futures[future]
                    failed_segments.append((segment, str(e)))
                    continue
        
        # Clear progress bars
        print("\n" * (len(segments) + 1))
        
        if failed_segments:
            print("\nSegment Processing Errors:")
            for segment, error in failed_segments:
                segment_path = os.path.basename(segment[0])
                print(f"\nSegment {segment_path}:")
                print(f"Error: {error}")
            raise Exception("Not all segments processed successfully")
        
        print("\nCombining results...")
        combine_videos(segment_results, output_prefix)
        combine_h5_files(segment_results, f"{output_prefix}_tracking.h5")
        
        print(f"\nProcessing complete. Final outputs:")
        print(f"  - {output_prefix}_original.mp4")
        print(f"  - {output_prefix}_masks.mp4")
        print(f"  - {output_prefix}_labels.mp4")
        print(f"  - {output_prefix}_tracking.h5")
        
    except Exception as e:
        print(f"\nError in parallel processing: {str(e)}")
        raise
    finally:
        # Clean up temp files
        for segment_path, _, _ in segments:
            temp_dir = os.path.dirname(segment_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # Clean up progress bars
        for pbar in progress_bars.values():
            pbar.close()

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